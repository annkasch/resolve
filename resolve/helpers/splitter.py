import torch
from sklearn.model_selection import train_test_split
import math

class Splitter():
    def __init__(self, mode="global", seed=None) -> None:
        super().__init__()
        self.mode = mode
        self.seed = seed
    
    def train_test_split(self,*arrays, groups, test_size=0.2, seed=None,):
        if self.mode == "batch_wise":
            return self.train_test_groupwise_split(*arrays, groups=groups, test_size=test_size, seed=self.seed)
        elif self.mode == "global":
            return train_test_split(*arrays, test_size=test_size, random_state=self.seed)
        else:
            return train_test_split(*arrays, test_size=test_size, random_state=self.seed)
        
    def train_test_groupwise_split(
            self,
            *arrays,
            groups,
            test_size=0.2,
            seed=None,
            return_indices=False,
            device=None,
            ensure_min_per_group=1,
        ):
            g = torch.as_tensor(groups, device=device)
            N = g.shape[0]
            gen = torch.Generator(device=device)
            if seed is not None:
                gen.manual_seed(seed)

            # unique groups and inverse map
            if g.ndim == 1:
                uniq, inv, counts = torch.unique(g, return_inverse=True, return_counts=True)
            else:
                uniq, inv, counts = torch.unique(g, dim=0, return_inverse=True, return_counts=True)

            G = uniq.shape[0]

            order = torch.argsort(inv)
            starts = torch.zeros(G + 1, dtype=torch.long, device=device)
            starts[1:] = torch.cumsum(counts, dim=0)

            # consistent clamp arguments
            k = (counts.float() * test_size).round().to(torch.long)
            k = torch.max(k, torch.full_like(k, ensure_min_per_group))
            k = torch.min(k, counts)

            test_chunks = []
            for i in range(G):
                s, e = starts[i].item(), starts[i + 1].item()
                c = e - s
                if k[i] == c:
                    test_chunks.append(order[s:e])
                else:
                    perm = torch.randperm(c, generator=gen, device=device)[:k[i]]
                    test_chunks.append(order[s + perm])

            test_idx = torch.sort(torch.cat(test_chunks))[0]
            keep = torch.ones(N, dtype=torch.bool, device=device)
            keep[test_idx] = False
            train_idx = torch.nonzero(keep, as_tuple=True)[0]

            if not arrays:
                arrays = (g,)

            out = []
            for arr in arrays:
                t = torch.as_tensor(arr, device=device)
                out.extend((t[train_idx], t[test_idx]))

            if return_indices:
                out.extend((train_idx, test_idx))
            return tuple(out)
    

    