from resolve.network_architectures import HCTargetAttnNP, ConditionalNeuralProcess, HCTargetAttnLNP
from resolve.network_architectures import Autoencoder, IsolationForestWrapper

class ModelsManager():
    def __init__(self, config):
        self.config = config
        
        self._models = {}
        self._factories = {
            "HCTargetAttnNP": lambda cfg: HCTargetAttnNP(cfg["d_theta"], cfg["d_phi"], cfg["d_y"], cfg.get("representation_size", 32)),
            "HCTargetAttnLNP": lambda cfg: HCTargetAttnLNP(cfg["d_theta"], cfg["d_phi"], cfg["d_y"], cfg.get("representation_size", 32)),
            "Autoencoder": lambda cfg: Autoencoder(cfg["d_theta"] + cfg["d_phi"], cfg.get("representation_size", 32), cfg.get("encoder_sizes", [128, 64])),
            "IsolationForest": lambda cfg: IsolationForestWrapper(cfg.get("n_estimators",200),cfg.get("contamination", 0.05), cfg.get("invert_scores", True)),
            "ConditionalNeuralProcess": lambda cfg: ConditionalNeuralProcess(cfg["d_theta"] + cfg["d_phi"]+cfg["d_y"], cfg.get("representation_size", 32), cfg.get("encoder_sizes", [128, 64]), cfg.get("decoder_sizes", [64, 128]), cfg["d_y"])
        }

    def get_network(self, model_name):
        if model_name not in self._models:
            if model_name not in self._factories:
                raise ValueError(f"Unknown model: {model_name}")
            self._models[model_name] = self._factories[model_name](self.config)
        return self._models[model_name]