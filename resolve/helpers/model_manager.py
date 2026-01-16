from resolve.conditional_neural_process_family import ConditionalNeuralProcess, AttnLNP, AttnCNP
from resolve.network_architectures import Autoencoder, VariationalAutoencoder, NeuralDensityRatioEstimator, FTTransformer
from resolve.network_architectures import InfoNCE, SupervisedContrastive, GNNBinaryClassifier
from resolve.network_architectures import IsolationForestWrapper, LightGBMWrapper, XGBoostWrapper
from resolve.network_architectures import NormalizingFlowClassifier
from resolve.dev import BDTFTTransformer, TreeConditionedCNP

class ModelsManager():
    def __init__(self, config, **kwargs):
        self.config = config
        config.update(kwargs)
        
        self._models = {}
        self._factories = {
            "AttnLNP": lambda cfg: AttnLNP(d_theta=cfg["d_theta"], d_phi=cfg["d_phi"], d_y=cfg["d_y"], d_model=cfg.get("representation_size", 32), encoder_hidden=cfg.get("encoder_sizes", [128, 128]), mode=cfg.get("mode", "concat"), theta_embed_dim=cfg.get("theta_embed_dim", None), n_heads=cfg.get("n_heads", 4), z_dim=cfg.get("z_dim", 0)),
            "AttnCNP": lambda cfg: AttnCNP(d_theta=cfg["d_theta"], d_phi=cfg["d_phi"], d_y=cfg["d_y"], d_model=cfg.get("representation_size", 32), encoder_hidden=cfg.get("encoder_sizes", [128, 128]), mode=cfg.get("mode", "concat"), theta_embed_dim=cfg.get("theta_embed_dim", None), n_heads=cfg.get("n_heads", 4)),
            "Autoencoder": lambda cfg: Autoencoder(cfg["d_theta"] + cfg["d_phi"], cfg.get("representation_size", 32), cfg.get("encoder_sizes", [128, 64])), 
            "VariationalAutoencoder": lambda cfg: VariationalAutoencoder(cfg["d_theta"] + cfg["d_phi"], cfg.get("representation_size", 32), cfg.get("encoder_sizes", [128, 64])),  
            "IsolationForest": lambda cfg: IsolationForestWrapper(cfg.get("n_estimators",100),cfg.get("max_samples",512),cfg.get("contamination","auto"),cfg.get("max_features",1.0),cfg.get("bootstrap",False),cfg.get("n_jobs",None),cfg.get("random_state",None),cfg.get("warm_star",False), cfg.get("invert_scores",True)),
            "ConditionalNeuralProcess": lambda cfg: ConditionalNeuralProcess(cfg["d_theta"] + cfg["d_phi"]+cfg["d_y"], cfg.get("representation_size", 32), cfg.get("encoder_sizes", [128, 64]), cfg.get("decoder_sizes", [64, 128]), cfg["d_y"], cfg.get("drop_out",0.)),
            "NeuralDensityRatioEstimator": lambda cfg: NeuralDensityRatioEstimator(d_theta=cfg["d_theta"],d_phi=cfg["d_phi"], d_y=cfg["d_y"], d_model=cfg.get("representation_size", 32), encoder_hidden=cfg.get("encoder_sizes", [128, 128]), decoder_hidden=cfg.get("decoder_sizes", [128, 128]), mode=cfg.get("mode", "concat"), theta_embed_dim=cfg.get("theta_embed_dim", None)),
            "SupervisedContrastive": lambda cfg: SupervisedContrastive(d_theta=cfg["d_theta"],d_phi=cfg["d_phi"], d_y=cfg["d_y"], d_model=cfg.get("representation_size", 32), encoder_hidden=cfg.get("encoder_sizes", [128, 128]), decoder_hidden=cfg.get("decoder_sizes", [128, 128]), d_proj=cfg.get("projection_size", 64), lambda_contrast=cfg.get("lambda_contrast", 0.01), mode=cfg.get("mode", "concat"), theta_embed_dim=cfg.get("theta_embed_dim", None)),
            "InfoNCE": lambda cfg: InfoNCE(d_theta=cfg["d_theta"],d_phi=cfg["d_phi"], d_y=cfg["d_y"], d_model=cfg.get("representation_size", 32), encoder_hidden=cfg.get("encoder_sizes", [128, 128]), decoder_hidden=cfg.get("decoder_sizes", [128, 128]), d_proj=cfg.get("projection_size", 64), lambda_contrast=cfg.get("lambda_contrast", 0.01), mode=cfg.get("mode", "concat"), theta_embed_dim=cfg.get("theta_embed_dim", None)),
            "FTTransformer": lambda cfg: FTTransformer(d_theta=cfg["d_theta"],d_phi=cfg["d_phi"], d_y=cfg["d_y"], d_model=cfg.get("representation_size", 64),  depth=cfg.get("depth", 1), n_heads=cfg.get("n_heads", 4), use_cls_token=cfg.get("use_cls_token", True)),
            "XGBoost": lambda cfg: XGBoostWrapper(config=cfg["config"], task=cfg.get("task","binary"), out_dim=cfg["d_y"], use_parameter_search=cfg.get("use_parameter_search",False), use_leaf_embeddings=cfg.get("use_leaf_embeddings",False)),
            "LightGBM": lambda cfg: LightGBMWrapper(config=cfg["config"], task=cfg.get("task","binary"), out_dim=cfg["d_y"], use_parameter_search=cfg.get("use_parameter_search",False), use_leaf_embeddings=cfg.get("use_leaf_embeddings",False)),
            "TreeConditionedCNP": lambda cfg: TreeConditionedCNP(d_theta=cfg["d_theta"], d_phi=cfg["d_phi"], d_y=cfg["d_y"], out_dim=cfg.get("out_dim",1), tree_config=cfg["tree_config"], d_model=cfg.get("representation_size", 32), encoder_hidden=cfg.get("encoder_sizes", [128, 128]), mode=cfg.get("mode", "concat"), theta_embed_dim=cfg.get("theta_embed_dim", None), n_heads=cfg.get("n_heads", 4)),
            "BDTFTTransformer": lambda cfg: BDTFTTransformer(d_theta=cfg["d_theta"], d_phi=cfg["d_phi"], d_y=cfg["d_y"], out_dim=cfg.get("out_dim",1), tree_config=cfg["tree_config"], d_model=cfg.get("representation_size", 64), depth=cfg.get("depth", 1), n_heads=cfg.get("n_heads", 4), threshold=cfg.get("threshold", [0.1, 0.9]), use_tokenizer=cfg.get("use_tokenizer", False), use_cls_token=cfg.get("use_cls_token", True)),
            "GNNBinaryClassifier": lambda cfg: GNNBinaryClassifier(d_theta=cfg["d_theta"], d_phi=cfg["d_phi"], d_y=cfg["d_y"], d_model=cfg.get("representation_size", 64), encoder_hidden=cfg.get("encoder_sizes", [128, 128]), gnn_hidden=cfg.get("gnn_hidden", [128, 128]), dropout=cfg.get("dropout", 0.5)),
            "NormalizingFlowClassifier": lambda cfg: NormalizingFlowClassifier(dim=cfg["d_theta"] + cfg["d_phi"], n_flow_layers=cfg.get("n_flow_layers", 4), hidden_dims=cfg.get("hidden_dims", [64, 64])),
        }

    def get_network(self, model_name):
        if model_name not in self._models:
            if model_name not in self._factories:
                raise ValueError(f"Unknown model: {model_name}")
            self._models[model_name] = self._factories[model_name](self.config)
        return self._models[model_name]