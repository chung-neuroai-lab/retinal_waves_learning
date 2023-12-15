import os
import random
import numpy as np
import torch

from yacs.config import CfgNode
from configs.default_configs import get_cfg_defaults, get_dataset_dict

def get_config(cfg_lst=None):
    # 1. Get default config in default_configs.py
    cfg = get_cfg_defaults()

    # 2. Override from command line
    if cfg_lst == None:
        raise ValueError("Please provide a config string to override the default config")
    else:
        cfg_lst = cfg_lst[0].split()
        cfg.merge_from_list(cfg_lst)
    
    # 3. determine dataset_path
    if cfg.DATA.DATASET_PATH == None:
        dataset_dict = get_dataset_dict()
        cfg.DATA.DATASET_PATH = dataset_dict[cfg.DATA.DATASET_NAME]

    # 4. determine shuffling type
    cfg = determine_shuffling(cfg)

    # 5. resolve projector dimension
    proj_dim_str = resolve_projector_dim(cfg.MODEL.PROJECTOR_DIMS)
    cfg.MODEL.PROJECTOR_DIMS = proj_dim_str

    # 6. fix labeling
    cfg = fix_labeling(cfg)
    return cfg

def fix_labeling(cfg):
    user = os.getenv("USER")
    
    cfg.DATA.RRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH = f"/mnt/home/{user}/ceph/retinal_waves_learning/data/watershed_waves_labels.npy"
    cfg.DATA.MRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH = f"/mnt/home/{user}/ceph/retinal_waves_learning/data/model_waves_labels.npy"

    return cfg

def resolve_projector_dim(proj_dim_str):
    proj_dim_str = list(map(int,proj_dim_str.split("-")))
    return proj_dim_str
    

def determine_shuffling(cfg):
    if cfg.RAND_INIT.LOAD_RAND_INIT_ENCODER == True:
        pass
    else:
        print(f"cfg.SSL_TRAIN.SHUFFLE_TYPE={cfg.SSL_TRAIN.SHUFFLE_TYPE}")
        if cfg.SSL_TRAIN.SHUFFLE_TYPE=="unshuffled":
            pass
        elif cfg.SSL_TRAIN.SHUFFLE_TYPE=="temporal_shuffling":
            cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_TEMPORAL_SHUFFLE = True
        elif cfg.SSL_TRAIN.SHUFFLE_TYPE=="spatial_shuffling":
            cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_SPATIAL_SHUFFLE = True
        elif cfg.SSL_TRAIN.SHUFFLE_TYPE=="spatial_temporal_shuffling":
            cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_TEMPORAL_SHUFFLE = True
            cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_SPATIAL_SHUFFLE = True
        else:
            raise ValueError

    return cfg


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def build_results_dir():
    curr_git_repo_dir = "./results"
    if not os.path.exists(curr_git_repo_dir):
        os.mkdir(curr_git_repo_dir)
    
    lst_of_data = [
        "CIFAR100",
        "CIFAR10",
        "MNIST",
        "real_retinal_waves",
        "real_retinal_waves_three_channels",
        "model_retinal_waves",
        "model_retinal_waves_three_channels",
        "real_retinal_waves_large",
        "real_retinal_waves_three_channels_large",
        "model_retinal_waves_large",
        "model_retinal_waves_three_channels_large",
    ]
    lst_of_model = ["SimCLR", "MMCR"]
    lst_of_seed = [42, 43, 44, 45, 46]
    for data in lst_of_data:
        if not os.path.exists(os.path.join(curr_git_repo_dir, data)):
            os.mkdir(os.path.join(curr_git_repo_dir, data))
        for model in lst_of_model:
            if not os.path.exists(os.path.join(curr_git_repo_dir, data, model)):
                os.mkdir(os.path.join(curr_git_repo_dir, data, model))
            for seed in lst_of_seed:
                if not os.path.exists(os.path.join(curr_git_repo_dir, data, model, f"seed_{seed}")):
                    os.mkdir(os.path.join(curr_git_repo_dir, data, model, f"seed_{seed}"))
