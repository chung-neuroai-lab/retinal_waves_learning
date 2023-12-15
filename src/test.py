import os
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader
import torch as ch
from torchvision import datasets, transforms
import torch.optim as optim

from src.data import get_dataset
from src.models import get_model
from src.losses import get_objective
from src.utils import seed_everything


def classifier_train_one_epoch(
    model, 
    train_dataloader,
    optimizer,
    objective,
    epoch,
    use_wandb,
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_train=True
    model.train()
    total_loss, total_correct_1, total_correct_5, total_num, data_bar = (
        0.0,
        0.0,
        0.0,
        0,
        tqdm(train_dataloader),
    )

    # objective = nn.CrossEntropyLoss()
    for data, target in data_bar:
        # breakpoint()
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss = objective(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += data.size(0)
        # breakpoint()
        total_loss += loss.item() * data.size(0)
        prediction = torch.argsort(out, dim=-1, descending=True)
        total_correct_1 += torch.sum(
            (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
        ).item()
        total_correct_5 += torch.sum(
            (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
        ).item()

        data_bar.set_description(
            "{} Epoch: [{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                "Train" if is_train else "Test",
                epoch,
                total_loss / total_num,
                total_correct_1 / total_num * 100,
                total_correct_5 / total_num * 100,
            )
        )

    if use_wandb:
        wandb.log({"train_epoch": epoch, "train_loss": total_loss / total_num, "train_acc1": total_correct_1 / total_num * 100, "train_acc5": total_correct_5 / total_num * 100}) #, "train_total_correct_1": total_correct_1, "train_total_correct_5": total_correct_5})

def classifier_eval_one_epoch(
    model,
    test_dataloader,
    objective,
    epoch,
    use_wandb,
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    is_train=False
    total_loss, total_correct_1, total_correct_5, total_num, data_bar = (
        0.0,
        0.0,
        0.0,
        0,
        tqdm(test_dataloader),
    )

    with torch.no_grad():
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = objective(out, target)

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()

            data_bar.set_description(
                "{} Epoch: [{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                    "Train" if is_train else "Test",
                    epoch,
                    total_loss / total_num,
                    total_correct_1 / total_num * 100,
                    total_correct_5 / total_num * 100,
                )
            )
        if use_wandb:
            wandb.log({"test_epoch": epoch, "test_loss": total_loss / total_num, "test_acc1": total_correct_1 / total_num * 100, "test_acc5": total_correct_5 / total_num * 100}) #, "test_total_correct_1": total_correct_1, "test_total_correct_5": total_correct_5})

    return total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def classifier_train(cfg):
    print(cfg)
    # === Device & Seed === #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['PYTHONUNBUFFERED'] = '1'
    seed_everything(cfg.SEED)

    # === Model === #
    if cfg.CLASSIFIER_TRAIN.PRETRAINED_PATH == None:
        pretrained_path_with_seed = None
    else:
        tmp_dir_lst = cfg.CLASSIFIER_TRAIN.PRETRAINED_PATH.split("/")
        tmp_dir_lst.insert(-1, f"seed_{cfg.SEED}")
        pretrained_path_with_seed = "/".join(tmp_dir_lst)

    if cfg.DATA.DATASET_NAME=="CIFAR100":
        num_class = 100
    elif cfg.DATA.DATASET_NAME=="CIFAR10":
        num_class = 10
    else:
        num_class = 10
        # raise NotImplementedError
    
    model = get_model(
        cfg.OBJECTIVE.OBJECTIVE_TYPE, 
        
        encoder=cfg.MODEL.ENCODER,
        # classifier arguments
        num_class=num_class,
        pretrain_dataset_name=cfg.CLASSIFIER_TRAIN.PRETRAINED_DATASET_NAME,
        pretrained_path=pretrained_path_with_seed,
        cfg=cfg
    )
    model = model.to(device)

    # === Dataset === #
    train_dataloader, test_dataloader = get_dataset(
        cfg.DATA.DATASET_NAME,
        cfg.DATA.DATASET_PATH,
        cfg.DATA.BATCH_SIZE,
        cfg.DATA.N_AUGS,
        cfg.DATA.USE_SHUFFLE,
        cfg.SYSTEM.CPUS_PER_TASK,

        # MNIST
        mnist_train_shuffle = cfg.MNIST.TRAIN_SHUFFLE,
        mnist_test_shuffle = cfg.MNIST.TEST_SHUFFLE,

        # CIFAR10
        cifar_train_shuffle = cfg.CIFAR10.TRAIN_SHUFFLE,
        cifar_test_shuffle = cfg.CIFAR10.TEST_SHUFFLE,

        #CIFAR100augs
        targets_path = cfg.CIFAR100_CLASSIFY_AUGS.TARGETS_PATH,
        aug_type = cfg.CIFAR100_CLASSIFY_AUGS.AUG_TYPE
        
    )

    # === Objective and Optimizer === #
    objective = get_objective(
        cfg.OBJECTIVE.OBJECTIVE_FUNC,
        cfg.DATA.BATCH_SIZE,
        cfg.DATA.N_AUGS,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.OPTIMIZER.BASE_LR, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
    )

    # === Classifier Training and Evaluation === #
    best_acc1 = 0
    best_acc5 = 0

    for epoch in range(1, cfg.CLASSIFIER_TRAIN.TOTAL_EPOCH):
        print(f"Current Epoch: {epoch}")

        classifier_train_one_epoch(
            model, 
            train_dataloader,
            optimizer,
            objective,
            epoch,
            cfg.WANDB.USE_WANDB,
        )

        acc1, acc5 = classifier_eval_one_epoch(
            model,
            test_dataloader,
            objective,
            epoch,
            cfg.WANDB.USE_WANDB,
        )
        if acc1 >= best_acc1:
            best_acc1 = acc1
        if acc5 >= best_acc5:
            best_acc5 = acc5


    # === Model Saving === #
    tmp_dir_lst = cfg.MODEL.MODEL_SAVE_PATH.split("/")
    tmp_dir_lst.insert(-1, f"seed_{cfg.SEED}")
    model_save_path_with_seed = "/".join(tmp_dir_lst)

    # torch.save(model.state_dict(), model_save_path_with_seed)
    print(f"model checkpoint saved to {model_save_path_with_seed}")

    print(f"best_acc1 = {best_acc1}%")
    print(f"best_acc5 = {best_acc5}%")

    try:
        test_acc_path = model_save_path_with_seed.replace("classifier", "test_acc")
        # torch.save([best_acc1,best_acc5], test_acc_path)
        print(f"test accuracy saved to {test_acc_path}")
    except Exception as e:
        print(f"Exception={e}")
