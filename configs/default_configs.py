from yacs.config import CfgNode

import os
user = os.getenv("USER")

# ----------------------------------------------------------------------------- #
# Config Node Initialization
# ----------------------------------------------------------------------------- #
_C = CfgNode()

# ***************************************************************************** #
# ***************************************************************************** #
# ***************************************************************************** #
# 1. General Configurations
# ***************************************************************************** #
# ***************************************************************************** #
# ***************************************************************************** #

# ----------------------------------------------------------------------------- #
# Objective
# ----------------------------------------------------------------------------- #
_C.OBJECTIVE = CfgNode()
# Available options: ["ssl_training", "classifier_training", "geometry_analysis"]
_C.OBJECTIVE.OBJECTIVE_TYPE = "ssl_training" 
# Available options: ["SimCLR", "CrossEntropy", "MFTMA"]
_C.OBJECTIVE.OBJECTIVE_FUNC = "SimCLR"

# ----------------------------------------------------------------------------- #
# Random Seed
# ----------------------------------------------------------------------------- #
# Available options: [42, 43, 44]
_C.SEED = 42

# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.MODEL_SAVE_PATH = "."
_C.MODEL.MODEL_SAVE_FILE_NAME = "model.pt"

# ----------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------- #
_C.DATA = CfgNode()
# Available options: ["real_retinal_waves", "real_retinal_waves_three_channels", 
# "model_retinal_waves", "model_retinal_waves_three_channels",
# "real_retinal_waves_large", "real_retinal_waves_three_channels_large"
# "model_retinal_waves_large", "model_retinal_waves_three_channels_large",
# "MNIST", "CIFAR10"]
_C.DATA.DATASET_NAME = "real_retinal_waves"

# ----------------------------------------------------------------------------- #
# System Configurations for SLURM Cluster Job Submission
# ----------------------------------------------------------------------------- #
_C.SYSTEM = CfgNode()
_C.SYSTEM.SLURM_FOLDER = f"/mnt/home/{user}/ceph/slurm/retinal_waves"

# ----------------------------------------------------------------------------- #
# Weights and Biases
# ----------------------------------------------------------------------------- #
_C.WANDB = CfgNode()
_C.WANDB.USE_WANDB = False
_C.WANDB.PROJECT = "retinal_waves_learning"
_C.WANDB.ENTITY = "retinal_waves_learning_proj"
_C.WANDB.NAME = "TODO_SET_NAME"
# wandb.init(project="retinal_waves_learning",name="proj_creation",entity="retinal_waves_learning_proj",config=None)
# TODO: Not yet implemented
_C.WANDB.SWEEP = CfgNode()
_C.WANDB.SWEEP.USE_SWEEP = False
_C.WANDB.SWEEP.SWEEP_CONFIG = ""
_C.WANDB.SWEEP.SWEEP_COUNTS = 10



# ***************************************************************************** #
# ***************************************************************************** #
# ***************************************************************************** #
# 2. Specific Configurations
# ***************************************************************************** #
# ***************************************************************************** #
# ***************************************************************************** #

# ----------------------------------------------------------------------------- #
# Random Init
# ----------------------------------------------------------------------------- #
_C.RAND_INIT = CfgNode()
_C.RAND_INIT.LOAD_RAND_INIT_ENCODER = False

# ----------------------------------------------------------------------------- #
# SSL Training
# ----------------------------------------------------------------------------- #
_C.SSL_TRAIN = CfgNode()
_C.SSL_TRAIN.TOTAL_EPOCH = 100 
_C.SSL_TRAIN.SAVE_EVERY = 0
_C.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES = True
# Available options: ["unshuffled", "temporal_shuffling", "spatial_shuffling", "spatial_temporal_shuffling"]
_C.SSL_TRAIN.SHUFFLE_TYPE=None
_C.SSL_TRAIN.FRAME_SEPARATION=1
_C.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS = False
_C.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_TEMPORAL_SHUFFLE = False
_C.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_SPATIAL_SHUFFLE = False

# ----------------------------------------------------------------------------- #
# Classifier Training
# ----------------------------------------------------------------------------- #
_C.CLASSIFIER_TRAIN = CfgNode()
_C.CLASSIFIER_TRAIN.PRETRAINED_DATASET_NAME = "real_retinal_waves"
_C.CLASSIFIER_TRAIN.PRETRAINED_SSL_OBJECTIVE = None
_C.CLASSIFIER_TRAIN.PRETRAINED_PATH = f"/mnt/home/{user}/ceph/retinal_waves_learning/results/real_retinal_waves/SimCLR/model.pt"
_C.CLASSIFIER_TRAIN.TOTAL_EPOCH = 100
_C.CLASSIFIER_TRAIN.SSL_LR = 0.001
_C.CLASSIFIER_TRAIN.USE_PROJECTOR = False
_C.CLASSIFIER_TRAIN.CLASSIFY_INTERMEDIATE_LAYER = False
_C.CLASSIFIER_TRAIN.CLASSIFY_LAYER_NUM = 4

# ----------------------------------------------------------------------------- #
# MFTMA Analysis
# ----------------------------------------------------------------------------- #
_C.MFTMA_ANALYSIS = CfgNode()
_C.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME = "real_retinal_waves"
_C.MFTMA_ANALYSIS.PRETRAINED_PATH = f"/mnt/home/{user}/ceph/retinal_waves_learning/results/real_retinal_waves/SimCLR/model.pt"
_C.MFTMA_ANALYSIS.SAVE_PATH = "/mnt/home/ykuang/ceph/retinal_waves_learning/results/real_retinal_waves/SimCLR/model.pt"
_C.MFTMA_ANALYSIS.SAMPLED_CLASSES_OR_EXEMPLARS = 10
_C.MFTMA_ANALYSIS.EXAMPLES_PER_CLASS_OR_EXEMPLARS = 100

_C.MFTMA_ANALYSIS.USE_SIM_CAPACITY = False
_C.MFTMA_ANALYSIS.USE_SPARSE_CAPACITY = False
_C.MFTMA_ANALYSIS.IMAGE_MANIFOLD_TYPE = "class"
_C.MFTMA_ANALYSIS.AUG_TYPE = "translation"

# ----------------------------------------------------------------------------- #
# MFTMA Analysis - Wave Capacity
# ----------------------------------------------------------------------------- #
_C.MFTMA_ANALYSIS.WAVES_CAPACITY = CfgNode()
_C.MFTMA_ANALYSIS.WAVES_CAPACITY.TEST_WAVES_CAPACITY = False
_C.MFTMA_ANALYSIS.WAVES_CAPACITY.WAVE_TIMEPOINTS_PATH = f"/mnt/home/{user}/ceph/retinal_waves_learning/data/watershed_waves_labels.npy"
# 14 is the lower cutoff. if 10000, then all the wave manifolds are used
_C.MFTMA_ANALYSIS.WAVES_CAPACITY.CUTOFF_WAVE_LENGTH = 10000 


# ***************************************************************************** #
# ***************************************************************************** #
# ***************************************************************************** #
# 3. Stable Configurations
# ***************************************************************************** #
# ***************************************************************************** #
# ***************************************************************************** #

# ----------------------------------------------------------------------------- #
# SSL Training - MMCR
# ----------------------------------------------------------------------------- #
# Available options: [True, False]
_C.OBJECTIVE.MMCR_IMPLICIT = False
# Available options: ["nuclear_norm", "dot_product"]
_C.OBJECTIVE.MMCR_LOCAL_COMPRESSION_METRIC = "dot_product"

# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
_C.MODEL.ENCODER = "resnet18" # using a much smaller backbone
# _C.MODEL.ENCODER = "resnet50"
_C.MODEL.LOAD_FROM_SAVED_RAND_INITIALIZATION = False
_C.MODEL.PROJECTOR_DIMS = "512-128" # [512, 128] -> using a much smaller projector dimensionality
# _C.MODEL.PROJECTOR_DIMS = [8192, 8192, 8192]


_C.MODEL.BIAS_LAST = False
_C.MODEL.ZERO_INIT_RESIDUAL = False

# ----------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------- #
# Available options: [
  # real retinal waves
    # "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/real_wave_tensor.pt",
    # "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/real_wave_tensor_three_channels.pt",
  # real retinal waves large
    # "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_real_data.pt",
    # "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_real_data_three_channels.pt", # TODO: Possibility of Contrastive Wave Learning
  # model retinal waves
    # "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/model_wave_tensor.pt",
    # "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/model_wave_tensor_three_channels.pt",
  # model retinal waves large
    # "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_model_data.pt",
    # "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_model_data_three_channels.pt",
  # MNIST 
    # TODO: fill in
  # CIFAR10
    # TODO: fill in
# "",
# ]
_C.DATA.DATASET_PATH = None
_C.DATA.CHW_DIM = [3,32,32]
_C.DATA.N_AUGS = 10
_C.DATA.BATCH_SIZE = 50
_C.DATA.BATCH_SIZE_LIMIT = 3000
_C.DATA.USE_VALIDATION = True
_C.DATA.TRAIN_PROP = 0.8 #.8 for model, .90 for train
# Remark: For the retinal wave dataset, use_shuffle implies destroying the temporal structures. 
_C.DATA.USE_SHUFFLE = False
# _C.DATA.RRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH = "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_real_data_control_labels_100k/large_area_real_data_control_labels_indices_all.npy"
_C.DATA.RRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH = f"/mnt/home/{user}/ceph/retinal_waves_learning/data/watershed_waves_labels.npy"
_C.DATA.MRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH = f"/mnt/home/{user}/ceph/retinal_waves_learning/data/model_waves_labels.npy"
# _C.DATA.MRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH = "/mnt/home/ykuang/ceph/retinal_waves_learning/data/model_wave_timepoints.npy"
_C.DATA.CUTOFF_WAVE_LENGTH = 10000


# ----------------------------------------------------------------------------- #
# Optimizer
# ----------------------------------------------------------------------------- #
_C.OPTIMIZER = CfgNode()
_C.OPTIMIZER.BASE_LR = 0.001
_C.OPTIMIZER.WEIGHT_DECAY = 0.000001
_C.OPTIMIZER.USE_LR_SCHEDULE = False
_C.OPTIMIZER.LR_DECAY_FACTOR = 1

# ----------------------------------------------------------------------------- #
# System Configurations for SLURM Cluster Job Submission
# ----------------------------------------------------------------------------- #
_C.SYSTEM.SLURM_MAX_NUM_TIMEOUT = 30
_C.SYSTEM.MEM_GB = 512
_C.SYSTEM.GPUS_PER_NODE = 1
_C.SYSTEM.TASKS_PER_NODE = 1
# Remark: number of works in the dataloader is always equal to the number of CPUs per task.
_C.SYSTEM.CPUS_PER_TASK = 10     
_C.SYSTEM.NODES = 1
_C.SYSTEM.TIMEOUT_MIN = 4096
_C.SYSTEM.SLURM_PARTITION = "gpu"
_C.SYSTEM.CONSTRAINT = "a100"
#try a100?
_C.SYSTEM.SLURM_ARRAY_PARALLELISM = 512

# ----------------------------------------------------------------------------- #
# Real / Model Retinal Waves Dataset
# ----------------------------------------------------------------------------- #
_C.REAL_RETINAL_WAVES = CfgNode()
_C.REAL_RETINAL_WAVES.LOAD_SAVED_REAL_WAVES = True
_C.REAL_RETINAL_WAVES.N_RETINA_NEURONS = 1024

# ----------------------------------------------------------------------------- #
# MNIST Dataset
# ----------------------------------------------------------------------------- #
_C.MNIST = CfgNode()
_C.MNIST.TRAIN_SHUFFLE = True
_C.MNIST.TEST_SHUFFLE = False

# ----------------------------------------------------------------------------- #
# CIFAR10 Dataset
# ----------------------------------------------------------------------------- #
_C.CIFAR10 = CfgNode()
_C.CIFAR10.TRAIN_SHUFFLE = True
_C.CIFAR10.TEST_SHUFFLE = False

# ----------------------------------------------------------------------------- #
# CIFAR100 Augs Classifier Dataset
# ----------------------------------------------------------------------------- #
_C.CIFAR100_CLASSIFY_AUGS = CfgNode()
_C.CIFAR100_CLASSIFY_AUGS.AUG_TYPE = "translation"
_C.CIFAR100_CLASSIFY_AUGS.TARGETS_PATH = f"/mnt/home/{user}/ceph/retinal_waves_learning/data/cifar-100-augs-targets.pt" 


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  return _C.clone()

def get_dataset_dict():
    user = os.getenv("USER")
    dataset_dict = {
        "real_retinal_waves":"/mnt/home/aligeralde/ceph/retinal_waves_learning/data/real_wave_tensor.pt", 
        "real_retinal_waves_three_channels":"/mnt/home/aligeralde/ceph/retinal_waves_learning/data/real_wave_tensor_three_channels.pt", 

        "model_retinal_waves":"/mnt/home/aligeralde/ceph/retinal_waves_learning/data/model_wave_tensor.pt", 
        "model_retinal_waves_three_channels":"/mnt/home/aligeralde/ceph/retinal_waves_learning/data/model_wave_tensor_three_channels.pt",

        "real_retinal_waves_large":"/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_real_data.pt", 
        "real_retinal_waves_three_channels_large":"/mnt/home/aligeralde/ceph/retinal_waves_learning/data/raw_waves.pt",
        
        "model_retinal_waves_large":"/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_model_data.pt",
        "model_retinal_waves_three_channels_large":"/mnt/home/aligeralde/ceph/retinal_waves_learning/data/model_waves.pt",
        # "model_retinal_waves_three_channels_large":"/mnt/home/ykuang/ceph/retinal_waves_learning/data/large_area_model_data_three_channels.pt",

        "CIFAR10":"/mnt/home/tyerxa/ceph/projects/mcmc/datasets",
        "CIFAR100":"/mnt/home/ykuang/ceph/retinal_waves_learning/data/cifar-100-python",
        "CIFAR100_classify_augs":f"/mnt/home/{user}/ceph/retinal_waves_learning/data/cifar-100-augs.pt"        
    }
    return dataset_dict

