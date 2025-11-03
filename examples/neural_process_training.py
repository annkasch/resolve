import torch
import torch.optim as optim
import os
from resolve.utilities import utilities as utils
from resolve.helpers import DataLoaderManager
from resolve.helpers import Trainer, ModelsManager
from resolve.helpers import AsymmetricFocalWithFPPenalty, log_prob, recon_loss_mse, skip_loss, bce_with_logits, brier
from torch.utils.tensorboard import SummaryWriter
import yaml
import json


def main():
    # Set the path to the yaml settings file here
    path_to_settings = "./binary-black-hole/"
    with open(f"{path_to_settings}/settings.yaml", "r") as f:
        config_file = yaml.safe_load(f)

    torch.manual_seed(0)
    version = config_file["path_settings"]["version"]
    path_out = f'{config_file["path_settings"]["path_out_model"]}/model-{version}'

    model_name = config_file["model_settings"]["network"]["model_used"]
    network_config = config_file["model_settings"]["network"]["models"][model_name]
    network_config["d_y"] = utils.get_feature_and_label_size(config_file)[1]
    network_config["d_theta"]  = len(config_file["simulation_settings"]["theta_labels"])
    network_config["d_phi"] = len(config_file["simulation_settings"]["phi_labels"])

    manager = ModelsManager(network_config)
    model = manager.get_network(config_file["model_settings"]["network"]["model_used"])


    # Total number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Memory in bytes (assuming float32 = 4 bytes per parameter)
    mem_bytes = num_params * 4
    mem_mb = mem_bytes / (1024 ** 2)
    mem_gb = mem_bytes / (1024 ** 3)

    print(f"Parameters: {num_params:,}")


    # load data:
    dataset_train = DataLoaderManager(mode = "train", 
                                    config_file=config_file
                                    )

    dataset_train.set_dataset(shuffle=config_file["model_settings"]["train"]["dataset"]["shuffle_dataset"])

    if config_file["model_settings"]["train"]["dataset"]["use_feature_normalization"] == "zscore":
        print("theta mean: ", dataset_train.dataset._normalizer._get_scaler("theta").mean_)
        print("phi mean: ", dataset_train.dataset._normalizer._get_scaler("phi").mean_)
    elif config_file["model_settings"]["train"]["dataset"]["use_feature_normalization"] == "minmax":
        print("theta mean: ", dataset_train.dataset._normalizer._get_scaler("theta").data_range_)
        print("phi mean: ", dataset_train.dataset._normalizer._get_scaler("phi").data_range_)

    os.system(f'mkdir -p {path_out}/model_{version}_tensorboard_logs')
    os.system(f'rm {path_out}/model_{version}_tensorboard_logs/events*')
    writer = SummaryWriter(log_dir=f'{path_out}/model_{version}_tensorboard_logs')

    optimizer = None if model.__class__.__name__ == "IsolationForestWrapper" else optim.Adam(model.parameters(), lr=config_file["model_settings"]["train"]["learning_rate"])

    # Instantiate the training wrapper for the first phase
    trainer = Trainer(model, dataset_train)

    trainer.epochs = config_file["model_settings"]["train"]["training_epochs"]

    trainer.criterion = AsymmetricFocalWithFPPenalty(
                alpha_pos=utils.get_nested(config_file, ["model_settings","train","loss","alpha_pos"], 1.),
                alpha_neg=utils.get_nested(config_file, ["model_settings","train","loss","alpha_neg"], 1.),
                gamma_pos=utils.get_nested(config_file, ["model_settings","train","loss","gamma_pos"], 0.),
                gamma_neg=utils.get_nested(config_file, ["model_settings","train","loss","gamma_neg"], 0.),
                lambda_fp=utils.get_nested(config_file, ["model_settings","train","loss","lambda_fp"],0.),
                tau_fp=utils.get_nested(config_file, ["model_settings","train","loss","tau_fp"],0.5),
                lambda_tp=utils.get_nested(config_file, ["model_settings","train","loss","lambda_tp"],0.),
                tau_tp=utils.get_nested(config_file, ["model_settings","train","loss","tau_tp"],0.5),
                reduction=utils.get_nested(config_file, ["model_settings","train","loss","reduction"], "mean"),
                base_loss_fn=globals()[utils.get_nested(config_file, ["model_settings","train","loss","base_loss_fn"], "bce_with_logits")],
            )

    # Train the model
    summary_train = trainer.fit(optimizer=optimizer, patience = config_file["model_settings"]["train"]["patience"], writer=writer, ckpt_dir=f"{path_out}/checkpoints", ckpt_name=f"model_{version}_best.pt",
            monitor="pr_auc", mode="max")

    torch.save(model.state_dict(), f'{path_out}/model_{version}_model.pth')


    dataset_train.dataset.close()
    writer.close()
    utils.cleanup_workspace({})

if __name__ == "__main__":
    main()