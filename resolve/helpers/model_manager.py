from resolve.network_architectures import HCTargetAttnNP, ConditionalNeuralProcess, HCTargetAttnLNP
from resolve.network_architectures import Autoencoder, IsolationForestWrapper, VariationalAutoencoder
from resolve.network_architectures.variational_auotencoder import VariationalAutoencoder
from resolve.network_architectures.marginalized_neural_ratio_estimator import MarginalizedNeuralRatioEstimator

class ModelsManager():
    def __init__(self, config):
        self.config = config
        
        self._models = {}
        self._factories = {
            "HCTargetAttnNP": lambda cfg: HCTargetAttnNP(cfg["d_theta"], cfg["d_phi"], cfg["d_y"], cfg.get("representation_size", 32)),
            "HCTargetAttnLNP": lambda cfg: HCTargetAttnLNP(cfg["d_theta"], cfg["d_phi"], cfg["d_y"], cfg.get("representation_size", 32)),
            "Autoencoder": lambda cfg: Autoencoder(cfg["d_theta"] + cfg["d_phi"], cfg.get("representation_size", 32), cfg.get("encoder_sizes", [128, 64])), 
            "VariationalAutoencoder": lambda cfg: VariationalAutoencoder(cfg["d_theta"] + cfg["d_phi"], cfg.get("representation_size", 32), cfg.get("encoder_sizes", [128, 64])),  
            "IsolationForest": lambda cfg: IsolationForestWrapper(cfg.get("n_estimators",100),cfg.get("max_samples",512),cfg.get("contamination","auto"),cfg.get("max_features",1.0),cfg.get("bootstrap",False),cfg.get("n_jobs",None),cfg.get("random_state",None),cfg.get("warm_star",False), cfg.get("invert_scores",True)),
            "ConditionalNeuralProcess": lambda cfg: ConditionalNeuralProcess(cfg["d_theta"] + cfg["d_phi"]+cfg["d_y"], cfg.get("representation_size", 32), cfg.get("encoder_sizes", [128, 64]), cfg.get("decoder_sizes", [64, 128]), cfg["d_y"]),
            "MarginalizedNeuralRatioEstimator": lambda cfg: MarginalizedNeuralRatioEstimator(d_theta=cfg["d_theta"],d_phi=cfg["d_phi"],theta_hidden_dims=cfg.get("encoder_sizes", [128, 64]),phi_hidden_dims=cfg.get("encoder_sizes", [128, 64]),head_hidden_dims=cfg.get("decoder_sizes", [128, 64]),dropout_p=cfg.get("dropout_p", 0.0)),
        }

    def get_network(self, model_name):
        if model_name not in self._models:
            if model_name not in self._factories:
                raise ValueError(f"Unknown model: {model_name}")
            self._models[model_name] = self._factories[model_name](self.config)
        return self._models[model_name]