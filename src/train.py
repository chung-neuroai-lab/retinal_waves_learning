import os
import wandb
import torch
import submitit
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data import get_dataset, handling_retinal_wave_shuffling
from src.models import get_model
from src.losses import get_objective
from src.utils import seed_everything, build_results_dir
from src.mftma.utils.make_manifold_data import make_wave_manifold_data

def SSL_pretrain_one_epoch(
    model,
    dataloader,
    epoch,
    optimizer,
    objective,
    SSL_batch_size,
    total_epoch,
    loss_lst,
):
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader)
    c = len(dataloader) * (epoch - 1)
    for step, img_batch in enumerate(train_bar, start=c):
        optimizer.zero_grad()
        features, out = model(img_batch.cuda(non_blocking=True))
        loss, loss_dict = objective(out)
        loss.backward()
        optimizer.step()
        total_num += SSL_batch_size
        total_loss += loss.item() * SSL_batch_size
        train_bar.set_description(
            "Train Epoch: [{}/{}] Loss: {:.4f} ".format(
                epoch,
                SSL_batch_size,
                total_loss / total_num,
            )
        )
        c+=1

    loss_lst.append(loss.item())

def SSL_pretrain_wave_manifolds_one_batch(
    model,
    wave_manifolds_data,
    lst_of_wave_length,
    epoch,
    optimizer,
    objective,
    total_epoch,
    loss_lst,
    use_wandb,
    batch_limit,
):
    model.train()
    
    # get batch size for retinal waves
    lst_of_batch = get_retinal_waves_batch_size(lst_of_wave_length, batch_size_threshold=batch_limit)
    lst_of_batch_size = [sum(lst_i) for lst_i in lst_of_batch]
    cumsum_lst_of_batch_size = np.cumsum(lst_of_batch_size)

    # forward and backward pass    
    for i, batch_size_i in enumerate(cumsum_lst_of_batch_size):
        print(f"Current Batch [{i}/{len(lst_of_batch_size)}]")
        if (i % 5) == 0:
            print(f"Current Batch [{i}/{len(lst_of_batch_size)}]")

        optimizer.zero_grad()  
        
        if i==0:
            features, out = model(wave_manifolds_data[0:batch_size_i])
        else:
            features, out = model(wave_manifolds_data[cumsum_lst_of_batch_size[i-1]:batch_size_i])

        
        loss, loss_dict = objective(out, curr_batch_wave_labels=lst_of_batch[i])
        loss.backward()
        optimizer.step()
    
    print(f"Train Epoch: [{epoch}/{total_epoch}] Loss: {loss.item()}")
    loss_lst.append(loss.item())

    if use_wandb:
        wandb.log({"ssl_loss": loss.item(), "epoch": epoch})

def get_retinal_waves_batch_size(lst_of_wave_length,batch_size_threshold=3000):
    left_ind = 0
    right_ind = 0

    lst_of_batch = []
    for i, wave_length in enumerate(lst_of_wave_length):
        right_ind = i
        if sum(lst_of_wave_length[left_ind:right_ind]) > batch_size_threshold and (sum(lst_of_wave_length[left_ind:right_ind]) % 4 ==0):
            curr_batch = lst_of_wave_length[left_ind:right_ind]
            lst_of_batch.append(curr_batch)
            left_ind = right_ind
            continue 
        
        if right_ind==(len(lst_of_wave_length)-1):
            curr_batch = lst_of_wave_length[left_ind:right_ind]
            lst_of_batch.append(curr_batch)
            break
    
    return lst_of_batch

def SSL_pretrain(cfg):
    print(cfg)
    # === Device & Seed & Path === #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['PYTHONUNBUFFERED'] = '1'
    user = os.getenv("USER")
    seed_everything(cfg.SEED)
    build_results_dir()

    # === Model === #
    model = get_model(
        cfg.OBJECTIVE.OBJECTIVE_TYPE,

        # model arguments
        encoder=cfg.MODEL.ENCODER,
        projector_dims=cfg.MODEL.PROJECTOR_DIMS,
        pretrain_dataset_name=cfg.DATA.DATASET_NAME,
        bias_last=cfg.MODEL.BIAS_LAST,
        zero_init_residual=cfg.MODEL.ZERO_INIT_RESIDUAL,

        # optional arguments
        load_from_saved_rand_initialization=cfg.MODEL.LOAD_FROM_SAVED_RAND_INITIALIZATION,
        cfg=cfg,
    )

    if cfg.MODEL.LOAD_FROM_SAVED_RAND_INITIALIZATION:
        pass
    else:
        rand_path = os.path.join(cfg.MODEL.MODEL_SAVE_PATH, f"seed_{cfg.SEED}", "initialization_" + cfg.MODEL.MODEL_SAVE_FILE_NAME)
        if not os.path.exists(rand_path):
            torch.save(model.state_dict(), rand_path)
            print(f"random initialized model saved to {rand_path}!")
        else:
            print(f"random initialized model exists! path: {rand_path}")
    
    model = model.to(device)

    # === Dataset === #
    train_dataloader, _ = get_dataset(
        cfg.DATA.DATASET_NAME,
        cfg.DATA.DATASET_PATH,
        cfg.DATA.BATCH_SIZE,
        cfg.DATA.N_AUGS,
        cfg.DATA.USE_SHUFFLE,
        # cfg.DATA.USE_VALIDATION,
        cfg.SYSTEM.CPUS_PER_TASK,

        # Real Retinal Waves
        load_saved_retinal_waves = cfg.REAL_RETINAL_WAVES.LOAD_SAVED_REAL_WAVES,
        n_retina_neurons = cfg.REAL_RETINAL_WAVES.N_RETINA_NEURONS,
    )

    # === Objective and Optimizer === #
    fixed_waves_size = False if cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES else True

    objective = get_objective(
        cfg.OBJECTIVE.OBJECTIVE_FUNC,
        cfg.DATA.BATCH_SIZE,
        cfg.DATA.N_AUGS,
        fixed_waves_size=fixed_waves_size,
        chop_variable_waves_size_into_segments=cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS,
        frame_separation=cfg.SSL_TRAIN.FRAME_SEPARATION,
        mmcr_implicit=cfg.OBJECTIVE.MMCR_IMPLICIT,
        local_compression_metric=cfg.OBJECTIVE.MMCR_LOCAL_COMPRESSION_METRIC,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.OPTIMIZER.BASE_LR, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
    )

    if cfg.OPTIMIZER.USE_LR_SCHEDULE == True:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.OPTIMIZER.LR_MILESTONES, gamma=cfg.OPTIMIZER.LR_DECAY_FACTOR)
        print(f"model | train_dataloader | objective | optimizer | lr_schedule = {model} | {train_dataloader} | {objective} | {optimizer} | {scheduler}")

    else:
        print(f"model | train_dataloader | objective | optimizer = {model} | {train_dataloader} | {objective} | {optimizer}")


    # === SSL Training === #
    loss_lst = []
    val_loss_lst = []

    if cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES:
        if cfg.DATA.DATASET_NAME == "real_retinal_waves_three_channels_large":
            wave_timepoints = np.load(cfg.DATA.RRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH)
        elif cfg.DATA.DATASET_NAME == "model_retinal_waves_three_channels_large":
            wave_timepoints = np.load(cfg.DATA.MRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH)
        else:
            raise NotImplementedError("No wave manifold labeling available!")
        
        if user == "aligeralde" or "ykuang":
            data_base_path = "/mnt/home/aligeralde/ceph/retinal_waves_learning/data"
            data_path = os.path.join(data_base_path, cfg.DATA.DATASET_NAME)
            seeded_data_path = os.path.join(data_path, f"seed_{cfg.SEED}")
            if not os.path.exists(data_path):
                os.mkdir(data_path)
                
            if not os.path.exists(seeded_data_path):
                os.mkdir(seeded_data_path)
        else:
            raise NotImplementedError

        cutoff_wave_length = cfg.DATA.CUTOFF_WAVE_LENGTH

        # transform wave_timepoints
        if cfg.DATA.DATASET_NAME == "real_retinal_waves_three_channels_large":
            pass
        
        if cfg.DATA.USE_VALIDATION == True:
            all_indices = np.arange(len(wave_timepoints))
            np.random.shuffle(all_indices)
            
            val_indices = np.sort(all_indices[:int(len(wave_timepoints)*(1-cfg.DATA.TRAIN_PROP))])
            val_wave_timepoints = wave_timepoints[val_indices]

            train_indices = np.sort(all_indices[int(len(wave_timepoints)*(1-cfg.DATA.TRAIN_PROP)):])
            wave_timepoints = wave_timepoints[train_indices]
            
            val_data = make_wave_manifold_data(cfg.DATA.DATASET_NAME, train_dataloader.dataset.retinal_image_dataset, val_wave_timepoints, cutoff_wave_length)
            val_data = [d.to(device) for d in val_data]
            val_wave_manifolds_data = torch.cat(val_data, dim=0)
            lst_of_val_wave_length = [len(wave_i) for wave_i in val_data]

            val_wave_manifolds_data = handling_retinal_wave_shuffling(
                use_temporal_shuffling=cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_TEMPORAL_SHUFFLE,
                use_spatial_shuffling=cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_SPATIAL_SHUFFLE,
                seeded_data_path=seeded_data_path,
                curr_dataset_name=cfg.DATA.DATASET_NAME,
                wave_manifolds_data=val_wave_manifolds_data,
            )

        data = make_wave_manifold_data(cfg.DATA.DATASET_NAME, train_dataloader.dataset.retinal_image_dataset, wave_timepoints, cutoff_wave_length)
        
        data = [d.to(device) for d in data]

        wave_manifolds_data = torch.cat(data,dim=0)
        
        
        lst_of_wave_length = [len(wave_i) for wave_i in data]
        
        wave_manifolds_data = handling_retinal_wave_shuffling(
            use_temporal_shuffling=cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_TEMPORAL_SHUFFLE,
            use_spatial_shuffling=cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_SPATIAL_SHUFFLE,
            seeded_data_path=seeded_data_path,
            curr_dataset_name=cfg.DATA.DATASET_NAME,
            wave_manifolds_data=wave_manifolds_data,
        )
        
        
        for epoch in range(cfg.SSL_TRAIN.TOTAL_EPOCH):
            print(f"Current Epoch: {epoch}")
            SSL_pretrain_wave_manifolds_one_batch(
                model,
                wave_manifolds_data,
                lst_of_wave_length,
                epoch,
                optimizer,
                objective,
                cfg.SSL_TRAIN.TOTAL_EPOCH,
                loss_lst,
                cfg.WANDB.USE_WANDB,
                cfg.DATA.BATCH_SIZE_LIMIT
            )

            if cfg.DATA.USE_VALIDATION == True:
                evaluate_val_loss_one_batch(
                    model,
                    val_wave_manifolds_data,
                    lst_of_val_wave_length,
                    epoch,
                    objective,
                    cfg.SSL_TRAIN.TOTAL_EPOCH,
                    val_loss_lst,
                    cfg.WANDB.USE_WANDB,
                    cfg.DATA.BATCH_SIZE_LIMIT
                )

            if cfg.OPTIMIZER.USE_LR_SCHEDULE == True:
                scheduler.step()
                if epoch in scheduler.milestones:
                    print(f"LEARNING RATE HAS BEEN DECREASED at {epoch} by {scheduler.gamma} to {scheduler.get_last_lr()}!")

            # save checkpoint every 50 epochs
            if cfg.SSL_TRAIN.SAVE_EVERY == 0:
                pass
            else:
                if epoch % cfg.SSL_TRAIN.SAVE_EVERY == 0:
                    ssl_trained_path = os.path.join(cfg.MODEL.MODEL_SAVE_PATH, f"seed_{cfg.SEED}", f"epoch{epoch}_"+cfg.MODEL.MODEL_SAVE_FILE_NAME)
                    torch.save(model.state_dict(), ssl_trained_path)
                    print(f"SSL trained model saved to {ssl_trained_path} @ epoch {epoch}!")

    else:
        for epoch in range(cfg.SSL_TRAIN.TOTAL_EPOCH):
            print(f"Current Epoch: {epoch}")
            SSL_pretrain_one_epoch(
                model,
                train_dataloader,
                epoch,
                optimizer,
                objective,
                cfg.DATA.BATCH_SIZE,
                cfg.SSL_TRAIN.TOTAL_EPOCH,
                loss_lst
            )   

    # === Model Saving === #
    ssl_trained_path = os.path.join(cfg.MODEL.MODEL_SAVE_PATH, f"seed_{cfg.SEED}", cfg.MODEL.MODEL_SAVE_FILE_NAME)
    torch.save(model.state_dict(), ssl_trained_path)
    print(f"SSL trained model saved to {ssl_trained_path}!")

    loss_ssl_trained_path = os.path.join(cfg.MODEL.MODEL_SAVE_PATH, f"seed_{cfg.SEED}", "loss_"+cfg.MODEL.MODEL_SAVE_FILE_NAME)
    torch.save(loss_lst, loss_ssl_trained_path)
    print(f"SSL Loss saved to {loss_ssl_trained_path}!")

def evaluate_val_loss_one_batch(
    model,
    wave_manifolds_data,
    lst_of_wave_length,
    epoch,
    # optimizer,
    objective,
    total_epoch,
    loss_lst,
    use_wandb,
    batch_limit
):
    model.eval()
    # get batch size for retinal waves
    lst_of_batch = get_retinal_waves_batch_size(lst_of_wave_length, batch_size_threshold=batch_limit)
    lst_of_batch_size = [sum(lst_i) for lst_i in lst_of_batch]
    cumsum_lst_of_batch_size = np.cumsum(lst_of_batch_size)
    batch_loss_list = []
    # forward and backward pass    
    for i, batch_size_i in enumerate(cumsum_lst_of_batch_size):        
        if i==0:
            features, out = model(wave_manifolds_data[0:batch_size_i])
        else:
            features, out = model(wave_manifolds_data[cumsum_lst_of_batch_size[i-1]:batch_size_i])

        loss, _ = objective(out, curr_batch_wave_labels=lst_of_batch[i])
        batch_loss_list.append(loss.item())
    batch_loss_avg = np.mean(np.array(batch_loss_list))
    print(f"Validation loss over one batch: {batch_loss_avg}")
    loss_lst.append(batch_loss_avg)

    if use_wandb:
        wandb.log({"val_loss": batch_loss_avg, "epoch": epoch})