from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import zarr
from numcodecs import Blosc

class ZarrPredWriter:
    """
    Handles all persistence of prediction-time arrays to a Zarr store.

    Layout:
      worker_{k}/mu        : (N_k,)
      worker_{k}/q_theta   : (N_k, q_theta_dim)
      worker_{k}/q_phi     : (N_k, q_phi_dim)
      worker_{k}/y_true    : (N_k,) or (N_k, y_dim)
      worker_{k}/file_start: (n_files_k,)
      worker_{k}/file_count: (n_files_k,)
      meta/theta           : (n_files_total, theta_size)
      meta/worker_of_file  : (n_files_total,)
    """
    def __init__(
        self,
        out_store: str,
        theta_size: int,
        dtype=np.float32,
        mu_chunks: int = 262_144,
        compressor: Optional[Blosc] = None,
        mode: str = "a",
    ):
        self.dtype = dtype
        self.theta_size = theta_size
        self.mu_chunks = int(mu_chunks)
        self.compressor = compressor or Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

        self.z = zarr.open(out_store, mode=mode)
        self.meta = self.z.require_group("meta")
        self.theta_ds = self._req_meta(
            name="theta",
            shape=(0, theta_size),
            chunks=(1024, theta_size),
            dtype=dtype,
        )
        self.who_ds = self._req_meta(
            name="worker_of_file",
            shape=(0,),
            chunks=(4096,),
            dtype=np.int32,
        )

        # Per-worker array handles cached here after first ensure()
        self._workers: Dict[int, Dict[str, zarr.Array]] = {}
        # Per-worker streaming state (file boundaries)
        self._state: Dict[int, Dict[str, float]] = {}

    def _req_meta(self, name, shape, chunks, dtype):
        if name in self.meta:
            return self.meta[name]
        return self.meta.zeros(name, shape=shape, chunks=chunks, dtype=dtype, compressor=self.compressor)

    def ensure_worker(self, wk: int, q_theta_dim: int, q_phi_dim: int, y_dim: int) -> Dict[str, zarr.Array]:
        """Create or fetch all arrays for a worker group."""
        if wk in self._workers:
            return self._workers[wk]

        g = self.z.require_group(f"worker_{wk}")

        def req(name, shape, chunks, dtype):
            if name in g:
                return g[name]
            return g.zeros(name, shape=shape, chunks=chunks, dtype=dtype, compressor=self.compressor)

        mu_arr = req("mu", shape=(0,), chunks=(self.mu_chunks,), dtype=self.dtype)

        qtheta_chunks = (max(1, self.mu_chunks // max(1, q_theta_dim)), q_theta_dim)
        qtheta_arr = req("q_theta", shape=(0, q_theta_dim), chunks=qtheta_chunks, dtype=self.dtype)

        qphi_chunks = (max(1, self.mu_chunks // max(1, q_phi_dim)), q_phi_dim)
        qphi_arr = req("q_phi", shape=(0, q_phi_dim), chunks=qphi_chunks, dtype=self.dtype)

        if y_dim == 1:
            y_arr = req("y_true", shape=(0,), chunks=(self.mu_chunks,), dtype=self.dtype)
        else:
            y_chunks = (max(1, self.mu_chunks // max(1, y_dim)), y_dim)
            y_arr = req("y_true", shape=(0, y_dim), chunks=y_chunks, dtype=self.dtype)

        fs_arr = req("file_start", shape=(0,), chunks=(4096,), dtype=np.int64)
        fc_arr = req("file_count", shape=(0,), chunks=(4096,), dtype=np.int64)

        self._workers[wk] = {
            "group": g, "mu": mu_arr, "q_theta": qtheta_arr, "q_phi": qphi_arr,
            "y_true": y_arr, "file_start": fs_arr, "file_count": fc_arr
        }
        if wk not in self._state:
            self._state[wk] = {"n": 0, "mu_sum": 0.0, "mu_sumsq": 0.0, "y_sum": 0.0, "cur_start": None}
        return self._workers[wk]

    def append_batch(
        self,
        wk: int,
        mu: np.ndarray,                 # (N,)
        q_theta: np.ndarray,            # (N, q_theta_dim)
        q_phi: np.ndarray,              # (N, q_phi_dim)
        y_true: np.ndarray,             # (N,) or (N, y_dim)
        y_dim: int,
    ) -> None:
        """Append one batch for a worker and update streaming stats."""
        W = self._workers[wk]
        n_add = int(mu.shape[0])
        assert q_theta.shape[0] == n_add and q_phi.shape[0] == n_add and y_true.shape[0] == n_add

        old = W["mu"].shape[0]
        # resize and write
        W["mu"].resize((old + n_add,))
        W["mu"][old:old + n_add] = mu.astype(self.dtype, copy=False)

        W["q_theta"].resize((old + n_add, q_theta.shape[1]))
        W["q_theta"][old:old + n_add, :] = q_theta.astype(self.dtype, copy=False)

        W["q_phi"].resize((old + n_add, q_phi.shape[1]))
        W["q_phi"][old:old + n_add, :] = q_phi.astype(self.dtype, copy=False)

        if y_dim == 1:
            y_store = y_true.reshape(-1).astype(self.dtype, copy=False)
            W["y_true"].resize((old + n_add,))
            W["y_true"][old:old + n_add] = y_store
        else:
            y_store = y_true.astype(self.dtype, copy=False)
            if y_store.ndim == 1:
                y_store = y_store.reshape(-1, y_dim)
            W["y_true"].resize((old + n_add, y_store.shape[1]))
            W["y_true"][old:old + n_add, :] = y_store

        # init start-of-file if first batch for current file
        st = self._state[wk]
        if st["cur_start"] is None:
            st["cur_start"] = old

        # update streaming stats
        st["n"] += n_add
        st["mu_sum"] += float(mu.sum())
        st["mu_sumsq"] += float((mu * mu).sum())
        if y_dim == 1:
            st["y_sum"] += float(y_store.sum())

    def finalize_file(self, wk: int, theta_row: np.ndarray) -> Tuple[float, float, float, int]:
        """
        Called once per completed file for worker wk.
        Appends file_start/file_count and meta (theta, worker_of_file).
        Returns (mu_mean, sigma_mean, y_mean_or_0, count).
        """
        W = self._workers[wk]
        st = self._state[wk]
        n = max(int(st["n"]), 1)

        mu_avg = st["mu_sum"] / n
        mu_var = max(st["mu_sumsq"] / n - mu_avg * mu_avg, 0.0)
        sigma_avg = mu_var ** 0.5
        y_mean = st["y_sum"] / n if "y_sum" in st else 0.0

        # push file boundaries
        fs_old = W["file_start"].shape[0]
        W["file_start"].resize((fs_old + 1,))
        W["file_count"].resize((fs_old + 1,))
        W["file_start"][fs_old] = int(st["cur_start"])
        W["file_count"][fs_old] = n

        # push meta
        tf_old = self.theta_ds.shape[0]
        self.theta_ds.resize((tf_old + 1, self.theta_size))
        self.theta_ds[tf_old, :] = theta_row.astype(self.dtype, copy=False)
        self.who_ds.resize((tf_old + 1,))
        self.who_ds[tf_old] = int(wk)

        # reset for next file
        self._state[wk] = {"n": 0, "mu_sum": 0.0, "mu_sumsq": 0.0, "y_sum": 0.0, "cur_start": None}
        return mu_avg, sigma_avg, y_mean, n

class ZarrPredReader:
    def __init__(self, store_path: str):
        self.z = zarr.open(store_path, mode="r")
        self.meta = self.z["meta"]

    def workers(self):
        return sorted(int(k.split("_")[1]) for k in self.z.group_keys() if k.startswith("worker_"))

    def get_worker_arrays(self, wk: int):
        g = self.z[f"worker_{wk}"]
        return {
            "mu": g["mu"][:],
            "q_theta": g["q_theta"][:],
            "q_phi": g["q_phi"][:],
            "y_true": g["y_true"][:],
            "file_start": g["file_start"][:],
            "file_count": g["file_count"][:],
        }

    def files_meta(self):
        return {
            "theta": self.meta["theta"][:],
            "worker_of_file": self.meta["worker_of_file"][:],
        }