import os
import wandb
import argparse
import submitit

from src.utils import get_config, convert_to_dict
from src.train import SSL_pretrain
from src.test import classifier_train
from src.mftma_analysis import MFTMA_Pipeline

def launch_job(cfg):
    if cfg.OBJECTIVE.OBJECTIVE_TYPE == "ssl_training":
        SSL_pretrain(cfg)
    elif cfg.OBJECTIVE.OBJECTIVE_TYPE == "classifier_training":
        classifier_train(cfg)
    elif cfg.OBJECTIVE.OBJECTIVE_TYPE == "geometry_analysis":
        MFTMA_Pipeline(cfg)
    else:
        raise NotImplementedError

def main(args):
    cfg = get_config(args.cfg_lst)

    if args.seed != None:
        cfg.SEED = args.seed
        print(f"cfg.SEED overrided to args.seed={args.seed}")

    if cfg.WANDB.USE_WANDB:
        if cfg.RAND_INIT.LOAD_RAND_INIT_ENCODER:
            cfg_dict = {}
        else:
            cfg_dict = convert_to_dict(cfg)
            
        wandb.init(
            project=cfg.WANDB.PROJECT,
            name=cfg.WANDB.NAME,
            entity=cfg.WANDB.ENTITY,
            config=cfg_dict,
        )

    launch_job(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_lst", 
        nargs='+',
        default=None, 
        help="YAML config override from command line",
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None, 
        choices=[42, 43, 44, 45, 46],
        help='Choose an integer from the list [42, 43, 44, 45, 46] as the random seed'
    )
    
    args = parser.parse_args()
    main(args)


