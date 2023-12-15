# Retinal Wave Learning

This repo contains the implementation for the NeurIPS 2023 UniReps Workshop paper [Unsupervised Learning on Spontaneous Retinal Activity Leads to Efficient Neural Representation Geometry](https://arxiv.org/abs/2312.02791)

## Installation
To install, navigate to the main directory and enter 

```
conda env create --file environment.yml
conda activate retinal_wave
pip install -e .
```

## Configurations Management
This repo uses [YACS](https://github.com/rbgirshick/yacs/tree/master) for model training configurations management. The default config file `configs/default_configs.py` is overridden by provided config string under `--cfg_lst` in the command line.


## Quick Start

### SimCLR SSL Pretraining of ResNet50 over Real Retinal Waves 
To pretrain a ResNet model on real retinal waves dataset, run

```bash
python3 scripts/run_experiments.py --cfg_lst "OBJECTIVE.OBJECTIVE_TYPE ssl_training OBJECTIVE.OBJECTIVE_FUNC SimCLR SEED 42 DATA.DATASET_NAME real_retinal_waves_three_channels_large DATA.TRAIN_PROP 0.9 DATA.BATCH_SIZE_LIMIT 3000 SSL_TRAIN.LEARN_CONTRASTIVE_WAVES True SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS True SSL_TRAIN.FRAME_SEPARATION 1 SSL_TRAIN.SHUFFLE_TYPE unshuffled SSL_TRAIN.TOTAL_EPOCH 100 OPTIMIZER.BASE_LR 0.0001 MODEL.ENCODER resnet9 MODEL.PROJECTOR_DIMS 8192-8192-8192 MODEL.MODEL_SAVE_PATH /mnt/home/ykuang/ceph/retinal_waves_learning/results/real_retinal_waves_three_channels_large/SimCLR MODEL.MODEL_SAVE_FILE_NAME TMP_resnet9_real_retinal_waves_three_channels_large_100epoch_wave_contrastive_chop_1frame_unshuffled_lr0.0001_projdim8192-8192-8192.pt WANDB.USE_WANDB False WANDB.PROJECT retinal_waves_learning_SSL WANDB.NAME SimCLR_42_resnet9_real_retinal_waves_three_channels_large_100epoch_wave_contrastive_chop_1frame_unshuffled_lr0.0001_projdim8192-8192-8192 SSL_TRAIN.SAVE_EVERY 0"
```

### Cross-Entropy Classifier Training of ResNet50 over MNIST
Run the following command for training a classifier for MNIST classification over fixed pretrained ResNet50 encoder weights: 

```
cd scripts/
python3 run_experiments.py --cfg [PATH_TO_GITHUB_REPO_FOLDER]/retinal_waves_learning/configs/MNIST/real_rw_pretrained_resnet50_linear_eval_MNIST_batch_size_100_n_augs_2.yaml
```

### Hyperparameter Sweeping in SSL & Classifier Training
Run all experiments with the following command:
```
cd scripts/
chmod +x submit_experiments.sh
submit_experiments.sh
```

### Mean-Field Theoretical Manifold Analysis (MFTMA)
Run all MFTMA analysis experiments with the following command:
```
cd scripts/
chmod +x submit_experiments.sh
submit_experiments.sh
```
