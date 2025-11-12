
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
import argparse
import multiprocessing as mp

def main(path_to_settings):
    # Set the path to the yaml settings file here
    with open(path_to_settings, "r") as f:
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

    model.memory_bank.build(dataset_train.dataset.data["train"]["theta"][0],dataset_train.dataset.data["train"]["phi"][0])

    print("memory build end")
    trainer.nepochs = config_file["model_settings"]["train"]["training_epochs"]

    if isinstance(utils.get_nested(config_file, ["model_settings","train","dataset","positive_ratio_train"], False), list):
            trainer.criterion = AsymmetricFocalWithFPPenalty(
                            alpha_pos=1.,
                            alpha_neg=1.,
                            gamma_pos=0.,
                            gamma_neg=0.,
                            lambda_fp=0.,
                            tau_fp=0.5,
                            lambda_tp= 5.,
                            tau_tp=0.5,
                            reduction=utils.get_nested(config_file, ["model_settings","train","loss","reduction"], "mean"),
                            base_loss_fn=globals()[utils.get_nested(config_file, ["model_settings","train","loss","base_loss_fn"], "bce_with_logits")],
                    )

            if utils.get_nested(config_file, ["model_settings","train","dataset","skip_warmup"], False) == True:
                    print("loading warm up")
                    model.load_state_dict(torch.load(f'{path_out}/model_{version}_warmup_model.pth'))
            else:
                    # Train the model
                    summary_train = trainer.warm_up(target_pos_frac = utils.get_nested(config_file, ["model_settings","train","dataset","positive_ratio_train"], None),
                            optimizer= optimizer,
                            writer=writer,
                            monitor = "pr_auc",
                            mode = "max",
                            save_best = True,
                            patience = 20,
                            num_data_pass_per_phase = utils.get_nested(config_file, ["model_settings","train","dataset","num_data_pass_per_phase"], 1.)
                    )

                    torch.save(model.state_dict(), f'{path_out}/model_{version}_warmup_model.pth')


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

    _ = trainer.evaluate(writer=writer, dataset_name="test")

    normalizer_train = dataset_train.dataset._normalizer

    # load data:
    dataset_test = DataLoaderManager(mode = "test", 
                                    config_file=config_file
                                    )
    dataset_test.set_dataset(normalizer=normalizer_train, shuffle=config_file["model_settings"]["train"]["dataset"]["shuffle_dataset"])

    tester = Trainer(model, dataset_test, epochs=1)
    tester._report = 1
    tester.criterion = trainer.criterion

    # Train the model
    summary_test = tester.evaluate(dataset_name="test", epoch=1, monitor="pr_auc",writer=writer)

    tester.metrics['test_2'] = tester.metrics.pop('test')
    trainer.metrics |= tester.metrics

    torch.save(model.state_dict(), f'{path_out}/model_{version}_model.pth')
    with open(f'{path_out}/model_{version}_settings.yaml', "w") as f:
        yaml.safe_dump(dataset_train.config_file, f)

    safe_metrics = utils.make_json_safe(trainer.metrics)

    with open(f'{path_out}/model_{version}_train_metrics.json', "w") as f:
        json.dump({model.__class__.__name__: safe_metrics}, f, indent=4)

    dataset_train.dataset.close()
    writer.close()
    utils.cleanup_workspace({})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, required=True)
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)  # critical on macOS
    main(args.settings)