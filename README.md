# Resolve

# Overview

This project tackles the challenge of building surrogate models under **rare-event** regimes, where the quantity of interest (e.g. background event rate in detector design or formation efficiency in binary black hole populations) is extremely small and sparsely sampled. It aims to replace massive, high-cost simulation campaigns with efficient surrogates that embed prior knowledge and manage high variance in the target metric. The result: enabling optimization or inference in scenarios where brute-force simulation is computationally infeasible (as in [**RESuM**](https://arxiv.org/pdf/2410.03873) and [**RESOLVE**](https://arxiv.org/pdf/2506.00757)).

The core concept combines **multi-fidelity modeling** and **probabilistic neural processes** to learn from both noisy low-fidelity data and scarce high-fidelity samples. The **neural process** targets noise reduction by learning structured correlations in the data and provides denoised outputs as input to the **multi-fidelity regression**, which in turn minimizes the overall **computational cost** while maintaining predictive accuracy and uncertainty awareness.

## Installation

For local development, install the `resolve` library:
```bash
pip install -e .
```

## Neural Process Network Training Guide

This guide explains how to train a neural network using the provided settings.yaml configuration and Jupyter notebook.

### Prerequisites

In addition to installing the resolve library, you'll need:
- Python 3.x
- PyTorch
- Jupyter Notebook
- Additional packages:
  ```bash
  pip install h5py pyyaml tensorboard sklearn
  ```

### Configuration Setup (Example)

1. Open `examples/binary-black-hole/settings.yaml` and configure the following sections:

#### Path Settings
```yaml
path_settings:
  version: v1.0.0  # Version identifier for your model
  path_to_files_train: ./in/data/lf/v1.0/training_cnp  # Path to training data
  path_out_model: ./out  # Output directory for model files
```

#### Model Settings

##### Network Architecture
```yaml
model_settings:
  network:
    model_used: ConditionalNeuralProcess  # Choose from: ConditionalNeuralProcess, HCTargetAttnNP, HCTargetAttnLNP
    models:
      DeterministicModel:
        representation_size: 32
        encoder_sizes: [32, 64, 64, 48]  # Encoder layer sizes
        decoder_sizes: [32, 64, 64, 48]  # Decoder layer sizes
```

##### Training Parameters
```yaml
  train:
    dataset:
      train_ratio: 0.6  # Ratio of data for training
      val_ratio: 0.2    # Ratio of data for validation
      test_ratio: 0.2   # Ratio of data for testing
      mixup_ratio: 0.5  # Ratio of samples for mixup augmentation
      use_beta: [0.1, 0.1]  # Beta distribution parameters for mixup
      context_ratio: 0.33  # Ratio of context points
      use_feature_normalization: zscore  # Normalization method: none, zscore, minmax
    
    batch_size: 1000
    training_epochs: 10
    learning_rate: 0.001
```

##### Loss Function Configuration
```yaml
    loss:
      base_loss_fn: log_prob    # Options: bce_with_logits, log_prob, mse
      alpha_pos: 1.0    # Weight for positive class
      alpha_neg: 1.0    # Weight for negative class
      gamma_pos: 2.0    # Focal loss parameter for positives
      gamma_neg: 2.0    # Focal loss parameter for negatives
```

### Training Process

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to and open `examples/neural_process_training.ipynb`

3. Execute each cell in sequence. The notebook will:
   - Load configurations from settings.yaml
   - Initialize the model architecture
   - Load and preprocess the training data
   - Train the model with specified parameters
   - Save the model and training metrics

4. Monitor training progress using TensorBoard:
   ```bash
   tensorboard --logdir=<path_out_model>/model_<version>_tensorboard_logs --host=0.0.0.0 --port=7007
   ```
   Access TensorBoard at: http://localhost:7007/

### Output Files

After training, you'll find the following files in your output directory:
- `model_<version>_model.pth`: Trained model weights
- `model_<version>_settings.yaml`: Copy of used settings
- `model_<version>_metrics.json`: Training metrics
- `model_<version>_tensorboard_logs/`: TensorBoard logs

### Memory Management

The training process includes automatic memory management:
- Dataset cleanup after training
- CUDA cache clearing (if using GPU)
- Garbage collection for unused objects

### Troubleshooting

If you encounter memory issues:
- Reduce batch_size in settings.yaml
- Monitor memory usage with:
  ```python
  from memory_profiler import profile
  ```
- Check GPU memory usage with:
  ```python
  torch.cuda.memory_summary()
  ```

For optimal performance:
- Ensure data paths are correctly set
- Verify normalization settings match your data
- Monitor training metrics in TensorBoard
