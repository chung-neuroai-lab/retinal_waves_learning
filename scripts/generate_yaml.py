'''
def yaml_generation_scripts(
    cfg: dict, 
    output_folder: str
):
    lst_of_cfg = []
    lst_of_output_filename = []

    # TODO: Implement Your YAML Generation Strategies Here
    assert len(lst_of_cfg) != 0, "TODO: Implement Your YAML Generation Strategies Here"
    assert len(lst_of_output_filename) != 0, "TODO: Implement Your YAML Generation Strategies Here"

    return lst_of_cfg, lst_of_output_filename
'''

import os
import copy
import argparse
from ruamel.yaml import YAML

def load_yaml(filename, yaml):
    with open(filename, 'r') as stream:
        try:
            return yaml.load(stream)
        except Exception as e:
            print(e)

def save_yaml(data, filename, yaml):
    with open(filename, 'w') as stream:
        try:
            yaml.dump(data, stream)
        except Exception as e:
            print(e)

def batch_generate_yaml(args):
    assert args.template_cfg_folder != None
    assert args.template_cfg_file != None

    # Initialize YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.boolean_representation = ['False', 'True']

    cfg = load_yaml(os.path.join(args.template_cfg_folder, args.template_cfg_file), yaml)
    lst_of_cfg, lst_of_output_filename = yaml_generation_scripts(cfg, args.template_cfg_folder)

    for i in range(len(lst_of_cfg)):
        save_yaml(lst_of_cfg[i], lst_of_output_filename[i], yaml)


# ==================================================================================================================================== #
# =============================================================== TODO =============================================================== #
# ==================================================================================================================================== #

# python3 scripts/generate_yaml.py \
#     --template_cfg_folder /mnt/home/aligeralde/ceph/retinal_waves_learning/configs/mftma/real_retinal_waves_three_channels_large \
#     --template_cfg_file mftma_real_rw_three_channels_large_pretrained_resnet50_linear_eval_CIFAR10_batch_size_100_n_augs_2.yaml

def yaml_generation_scripts(
    cfg: dict, 
    output_folder: str
):
    # TODO: use this function to batch generate YAML files
    lst_of_cfg = []
    lst_of_output_filename = []

    lst_of_augs = [2, 4, 10, 20, 40, 60]
    for augs in lst_of_augs:
        cfg_copy = copy.deepcopy(cfg)

        # TODO: Item 1
        cfg_copy['MFTMA_ANALYSIS']['PRETRAINED_PATH'] = f"/mnt/home/aligeralde/ceph/retinal_waves_learning/results/CIFAR10/SimCLR/classifier_over_shuffled_rw_three_channels_large_SSL_model_batch_size_100_n_augs_{augs}.pt"

        # TODO: Item 2
        output_filename = os.path.join(output_folder,f"mftma_shuffled_real_rw_three_channels_large_pretrained_resnet50_linear_eval_CIFAR10_batch_size_100_n_augs_{augs}.yaml")

        lst_of_cfg.append(cfg_copy)
        lst_of_output_filename.append(output_filename)

    return lst_of_cfg, lst_of_output_filename

# ==================================================================================================================================== #
# =============================================================== TODO =============================================================== #
# ==================================================================================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template_cfg_folder", 
        type=str, 
        default=None, 
        help="YAML config file folder",
    )
    parser.add_argument(
        "--template_cfg_file", 
        type=str, 
        default=None, 
        help="YAML config file name",
    )

    args = parser.parse_args()
    batch_generate_yaml(args)

