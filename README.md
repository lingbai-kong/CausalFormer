# PyTorch Implementation of CauseFormer: An Interpretable Transformer for Temporal Causal Discovery

Official PyTorch implementation for CauseFormer: An Interpretable Transformer for Temporal Causal Discovery.

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch (tested with PyTorch 1.11.0)
* Optional: CUDA (tested with CUDA 11.3)
* networkx
* numpy
* pandas
* scikit_learn

## Folder Structure
  ```
  CauseFormer/
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  ├── config/ - holds configuration for training
  │   ├── config_basic_diamond_mediator.json
  │   ├── config_basic_v_fork.json
  │   ├── config_fMRI.json
  │   └── config_lorenz.json
  ├── data/ - default directory for storing input data
  │   ├── basic
  │   ├── fMRI
  │   └── lorenz96
  ├── data_loader/
  ├── evaluator/
  ├── experiments.ipynb
  ├── explainer
  │   └── explainer.py
  ├── interpret.py - main script to start interpreting
  ├── LICENSE
  ├── logger/
  ├── model/ - models, relevance propogation, losses, and metrics
  │   ├── loss.py
  │   ├── metric.py
  │   ├── model.py
  │   ├── NonParamRP.py
  │   └── RRP.py
  ├── parse_config.py
  ├── README.md
  ├── requirements.txt
  ├── runner.py - integrated script to start running CauseFormer
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard
  ├── trainer/
  ├── train.py - main script to start training
  └── utils
  ```

## Usage

Try `python runner.py -c config/config_fMRI.json -t demo` to run code.

Checking experiments.ipynb for more experiments running.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Causality Learning", // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "PredictModel",     // name of model architecture to train
    "args": {
      "d_model": 512,           // Dimension of the embedding vector. D_QK in paper
      "n_head": 8,              // Number of attention heads. h in paper
      "n_layers": 1,            // single transformer encoder layer
      "ffn_hidden": 512,        // Hidden dimension in the feed forward layer. d_FFN in paper
      "drop_prob": 0,           // Dropout probability (Not used in practice)
      "tau": 10                 // Temperature hyperparameter for attention softmax
    }                
  },
  "data_loader": {
    "type": "TimeseriesDataLoader",    // selecting data loader
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 64,                // batch size
      "time_step": 32,                 // input window size. T in paper
      "output_window": 31,             // output window size
      "feature_dim": 1,                // input feature dim
      "output_dim": 1,                 // output window size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "masked_mse_torch",          // loss
  "metrics": [
    "accuracy", "masked_mse_torch"     // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
    "lam": 5e-4,                       // the coefficient for normalization
    "tensorboard": true,               // enable tensorboard visualization
  },
  "explainer": {
      "m":2,                           // number of top clusters of causal scores to consider.
      "n":3                            // number of total clusters for k-means clustering.
  }
}
```

## License
This project is licensed under the  GPL-3.0 License. See LICENSE for more details
This project is based on the [pytorch-template](https://github.com/victoresque/pytorch-template) GitHub template.
