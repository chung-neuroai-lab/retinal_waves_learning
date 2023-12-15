import sys
sys.path.append("/mnt/home/aligeralde/ceph/retinal_waves_learning")
import warnings
import traceback

import os
import wandb
import torch
import numpy as np
user = os.getenv("USER")

from src.utils import get_config, get_dataset_dict
from src.data import get_dataset
from src.models import get_model
from src.losses import get_objective

from src.mftma.utils.make_manifold_data import make_manifold_data, make_wave_manifold_data
from src.mftma.utils.activation_extractor import extractor
from src.mftma.manifold_analysis_correlation import manifold_analysis_corr
from src.mftma.manifold_analysis import manifold_analysis
from src.mftma.manifold_simcap_analysis import manifold_simcap_analysis
from src.mftma.simulated_cap import bisection_search

from src.mftma.alldata_dimension_analysis import alldata_dimension_analysis
from src.utils import seed_everything

def activations_extraction(sampled_classes_or_exemplars, model, data):
    # Extract activations
    print("starting activations extraction")
    activations = extractor(model, data, layer_types=['ReLU', 'Linear'])
    print(f"activations.keys() = {list(activations.keys())}")

    # Fix the incorrect tree traversal
    for layer, data, in activations.items():
        print("\n +++ Layer: {} +++ ".format(layer))
        print(f"len(data)={len(data)}")

        if len(data) == sampled_classes_or_exemplars*2:
            new_data = []
            
            for i in range(1, sampled_classes_or_exemplars*2, 2):
                new_data.append(data[i])        
            
            activations[layer] = new_data

        if len(data) == sampled_classes_or_exemplars*3:
            new_data = []
            
            for i in range(2, sampled_classes_or_exemplars*3, 3):
                new_data.append(data[i])        
            
            activations[layer] = new_data

    
    # check to see if the issue is fixed
    for layer, data, in activations.items():
        print("\n +++ Layer: {} +++ ".format(layer))
        print(f"len(data)={len(data)}")

        # for data_i in data:
        #     print(f"data_i.shape={data_i.shape}")

    # Projection based on JL-Lemma for Reduced Time Complexity
    print("Start Projection")

    for layer, data, in activations.items():
        X = [d.reshape(d.shape[0], -1).T for d in data]
        # Get the number of features in the flattened data
        N = X[0].shape[0]
        # If N is greater than 5000, do the random projection to 5000 features
        if N > 5000:
            print("Projecting {}".format(layer))
            M = np.random.randn(5000, N)
            M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
            print(f"M.shape = {M.shape}")
            X = [np.matmul(M, d) for d in X]
        activations[layer] = X
        
    return activations

def MFTMA_Pipeline(cfg):
    print(cfg)
    
    # === Device & Seed === #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['PYTHONUNBUFFERED'] = '1'
    seed_everything(cfg.SEED)

    if cfg.MFTMA_ANALYSIS.PRETRAINED_PATH == None: #random network
        pretrained_path = None
    else:
        # lst_path = cfg.MFTMA_ANALYSIS.PRETRAINED_PATH.split("/")
        # lst_path.insert(-1, f"seed_{cfg.SEED}")
        # pretrained_path = "/".join(lst_path)
        pretrained_path = cfg.MFTMA_ANALYSIS.PRETRAINED_PATH

    # === Model === #
    model = get_model(
        cfg.OBJECTIVE.OBJECTIVE_TYPE,

        # model arguments
        encoder=cfg.MODEL.ENCODER,
        projector_dims=cfg.MODEL.PROJECTOR_DIMS,
        pretrain_dataset_name=cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME,
        bias_last=cfg.MODEL.BIAS_LAST,
        zero_init_residual=cfg.MODEL.ZERO_INIT_RESIDUAL,
        
        # saved classifier checkpoints
        pretrained_path=pretrained_path,
    )
    model = model.to(device)
    model.eval()

    # === Dataset === #
    sampled_classes_or_exemplars=cfg.MFTMA_ANALYSIS.SAMPLED_CLASSES_OR_EXEMPLARS
    examples_per_class_or_exemplars=cfg.MFTMA_ANALYSIS.EXAMPLES_PER_CLASS_OR_EXEMPLARS

    if cfg.MFTMA_ANALYSIS.WAVES_CAPACITY.TEST_WAVES_CAPACITY:
        dataset_dict = get_dataset_dict()
        dataset = torch.load(dataset_dict[cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME])
        dataset = dataset.float()
        print(f"{dataset_dict[cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME]} loaded for wave manifolds")

        if cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME == "real_retinal_waves_three_channels_large":
            wave_timepoints = np.load(cfg.DATA.RRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH)
        elif cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME == "model_retinal_waves_three_channels_large":
            wave_timepoints = np.load(cfg.DATA.MRW_THREE_CHANNELS_LARGE_WAVE_TIMEPOINTS_PATH)
        else:
            raise NotImplementedError

        cutoff_wave_length = cfg.MFTMA_ANALYSIS.WAVES_CAPACITY.CUTOFF_WAVE_LENGTH

        data = make_wave_manifold_data(cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME, dataset, wave_timepoints, cutoff_wave_length)

        # breakpoint() 
   

        length_of_waves = [len(data_i) for data_i in data]
        # indices_of_selected_waves = np.where(np.array(length_of_waves)>=50)[0]
        indices_of_selected_waves = np.where(np.logical_and(np.array(length_of_waves)>=50, np.array(length_of_waves)<=132))[0][:50]
        # if cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME == "real_retinal_waves_three_channels_large":
            # indices_of_selected_waves = np.where(np.logical_and(np.array(length_of_waves)>=50, np.array(length_of_waves)<=132))[0]
        #     indices_of_selected_waves = np.where(np.logical_and(np.array(length_of_waves)>=50)[0]
        # elif cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME == "model_retinal_waves_three_channels_large":
        #     # indices_of_selected_waves = np.where(np.logical_and(np.array(length_of_waves)>=50, np.array(length_of_waves)<=106))[0]
        #     indices_of_selected_waves = np.where(np.logical_and(np.array(length_of_waves)>=50, np.array(length_of_waves)<=106))[0]
        # else:
        #     raise NotImplementedError

        # breakpoint()

        data = [data[ind] for ind in indices_of_selected_waves]
        data = [d.to(device) for d in data]
        print(f"wave manifold! length equals to {len(data)}")

        # TODO: define activation save path
        activation_save_path =f"/mnt/home/{user}/ceph/retinal_waves_learning/results/{cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME}/SimCLR/seed_{cfg.SEED}/{cfg.MFTMA_ANALYSIS.SAVE_PATH}" + f'_{cfg.MFTMA_ANALYSIS.AUG_TYPE}_activations.pt'

        # TODO: check activation saving & loading logic here
        if not os.path.exists(activation_save_path):
            activations = activations_extraction(len(data), model, data)
            torch.save(activations, activation_save_path)
            print("activations saved!")
            wandb.finish()
            sys.exit()
        else:
            activations = torch.load(activation_save_path)
            print("activations loaded!")

    else:
        if cfg.DATA.DATASET_NAME=="CIFAR100" and cfg.MFTMA_ANALYSIS.IMAGE_MANIFOLD_TYPE=="exemplar":
            exemplar_images_aug_transform = True
        else:
            exemplar_images_aug_transform = False
            
        train_dataloader, test_dataloader = get_dataset(
            cfg.DATA.DATASET_NAME,
            cfg.DATA.DATASET_PATH,
            cfg.DATA.BATCH_SIZE,
            cfg.DATA.N_AUGS,
            cfg.DATA.USE_SHUFFLE,
            cfg.SYSTEM.CPUS_PER_TASK,

            # CIFAR10
            cifar_train_shuffle = cfg.CIFAR10.TRAIN_SHUFFLE,
            cifar_test_shuffle = cfg.CIFAR10.TEST_SHUFFLE,

            # CIFAR100 augmentation transform
            exemplar_images_aug_transform=exemplar_images_aug_transform,
            n_transform=examples_per_class_or_exemplars,
            aug_type = cfg.MFTMA_ANALYSIS.AUG_TYPE
        )
        # breakpoint()
        # cfg.MFTMA_ANALYSIS.IMAGE_MANIFOLD_TYPE
        data = make_manifold_data(test_dataloader.dataset, sampled_classes_or_exemplars, examples_per_class_or_exemplars, seed=cfg.SEED, dataset_name=cfg.DATA.DATASET_NAME, manifold_type=cfg.MFTMA_ANALYSIS.IMAGE_MANIFOLD_TYPE)
        data = [d.to(device) for d in data]
        print(f"{cfg.DATA.DATASET_NAME} {cfg.MFTMA_ANALYSIS.IMAGE_MANIFOLD_TYPE} manifold!")

        # TODO: define activation save path
        activation_save_path =f"/mnt/home/{user}/ceph/retinal_waves_learning/results/{cfg.MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME}/SimCLR/seed_{cfg.SEED}/{cfg.WANDB.NAME}" + f'_{cfg.MFTMA_ANALYSIS.AUG_TYPE}_activations.pt'

        # TODO: check activation saving & loading logic here
        if not os.path.exists(activation_save_path):
            activations = activations_extraction(sampled_classes_or_exemplars, model, data)
            torch.save(activations, activation_save_path)
            print("activations saved!")
            wandb.finish()
            sys.exit()
        else:
            activations = torch.load(activation_save_path)
            print("activations loaded!")

        # if not os.path.exists(activation_save_path):
        #     activations = activations_extraction(sampled_classes_or_exemplars, model, data)
        #     # torch.save(activations, activation_save_path)
        #     # print("activations saved!")
        # else:
        #     activations = activations_extraction(sampled_classes_or_exemplars, model, data)
        #     # torch.save(activations, activation_save_path)
        #     # print("activations saved!")

        # if not os.path.exists(activation_save_path):
        #     activations = activations_extraction(sampled_classes_or_exemplars, model, data)
        #     torch.save(activations, activation_save_path)
        #     print("activations saved!")
        # else:
        #     activations = torch.load(activation_save_path)
        #     print("activations loaded!")
    
    
    # Start gathering theoretical manifold capacity measures
    capacities = []
    radii = []
    dimensions = []
    correlations = []
    lst_D_participation_ratio = []
    lst_D_explained_variance = []
    lst_D_feature = []

    # Start gathering simulated manifold capacity measures
    lst_of_asim = []
    lst_of_P = []
    lst_of_Nc0 = []
    lst_of_N_vec = []
    lst_of_p_vec = []

    lst_of_n_ = []

    mftma_num_repeat_calc = 1 # CHANGED TO 5 TO SPEED UP
    
    print("Start gathering manifold measures")
    for k, X, in activations.items():
        print(f"Current Layer -> {k}", flush=True)
        if cfg.MFTMA_ANALYSIS.USE_SIM_CAPACITY:
            if cfg.MODEL.ENCODER == "resnet50":
                if cfg.MODEL.PROJECTOR_DIMS==[512,128]:
                    raise NotImplementedError
                elif cfg.MODEL.PROJECTOR_DIMS==[8192, 8192, 8192]:
                    # ? 8192-8192-8192 proj_dim
                    # activations.keys() = ['layer_0_Input', 'layer_3_ReLU', 'layer_11_ReLU', 'layer_20_ReLU', 'layer_27_ReLU', 'layer_34_ReLU', 'layer_43_ReLU', 
                    # 'layer_50_ReLU', 'layer_57_ReLU', 'layer_64_ReLU', 'layer_73_ReLU', 'layer_80_ReLU', 'layer_87_ReLU', 'layer_94_ReLU', 'layer_101_ReLU', 'layer_108_ReLU', 'layer_117_ReLU', 
                    # 'layer_124_ReLU', 'layer_127_Linear', 'layer_129_ReLU', 'layer_130_Linear', 'layer_132_ReLU', 'layer_133_Linear']
                    sim_cap_layer_lst = ['layer_124_ReLU', 'layer_127_Linear', 'layer_129_ReLU', 'layer_130_Linear', 'layer_132_ReLU', 'layer_133_Linear']
                else:
                    raise NotImplementedError
            elif cfg.MODEL.ENCODER == "resnet18":
                # ? 8192-8192-8192 proj_dim
                # odict_keys(['layer_0_Input', 'layer_3_ReLU', 'layer_7_ReLU', 'layer_12_ReLU', 'layer_17_ReLU', 'layer_24_ReLU', 'layer_29_ReLU', 
                # 'layer_36_ReLU', 'layer_41_ReLU', 
                # 'layer_48_ReLU', 'layer_53_Linear', 'layer_55_ReLU', 'layer_56_Linear', 'layer_58_ReLU', 'layer_59_Linear'])
                # sim_cap_layer = "layer_48_ReLU"
                
                # ? 512-128 proj_dim
                # odict_keys(['layer_0_Input', 'layer_3_ReLU', 'layer_7_ReLU', 'layer_12_ReLU', 'layer_17_ReLU', 'layer_24_ReLU', 'layer_29_ReLU', 
                # 'layer_36_ReLU', 'layer_41_ReLU', 
                # 'layer_48_ReLU', 'layer_53_Linear', 'layer_55_ReLU', 'layer_56_Linear'])
                if cfg.MODEL.PROJECTOR_DIMS==[512,128]:
                    sim_cap_layer_lst = ['layer_48_ReLU', 'layer_53_Linear', 'layer_55_ReLU', 'layer_56_Linear']
                elif cfg.MODEL.PROJECTOR_DIMS==[8192, 8192, 8192]:
                    sim_cap_layer_lst = ['layer_48_ReLU', 'layer_53_Linear', 'layer_55_ReLU', 'layer_56_Linear', 'layer_58_ReLU', 'layer_59_Linear']
                else:
                    raise NotImplementedError
            elif cfg.MODEL.ENCODER == "resnet9":
                if cfg.MODEL.PROJECTOR_DIMS==[8192, 8192, 8192]:
                    sim_cap_layer_lst = ['layer_0_Input', 'layer_3_ReLU', 'layer_6_ReLU', 'layer_12_ReLU', 'layer_15_ReLU', 'layer_19_ReLU', 'layer_25_ReLU']
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            if k in sim_cap_layer_lst:
                lst_of_repeated_asim = []
                lst_of_repeated_P = []
                lst_of_repeated_Nc0 = []
                lst_of_repeated_N_vec = []
                lst_of_repeated_p_vec = []
                lst_of_repeated_n_ = []

                try:
                    if cfg.MFTMA_ANALYSIS.USE_SPARSE_CAPACITY:
                        split_=1/sampled_classes_or_exemplars
                        rep_=sampled_classes_or_exemplars
                    else:
                        split_=1/2
                        rep_=100 # 2^p = 2^10 = 1024. For computational complexity, we reduce it to just 100.

                    # simulated capacity -> func1      
                    # n_,ret_n_arr_,ret_prob_arr_, margin_arr_ = bisection_search(min_n=2,max_n=X[0].shape[0],rep=rep_,p=sampled_classes_or_exemplars,activations=X,split=split_)
                    # simulated capacity -> func2
                    asim0, P_, Nc0, N_vec, p_vec = manifold_simcap_analysis(X, n_rep=100)
                except Exception as e1:
                    print(f"yilun | e ={e1}")
                    traceback.print_exc()
                    asim0, P_, Nc0, N_vec, p_vec = 10000,10000,10000,10000,10000
                    # n_,ret_n_arr_,ret_prob_arr_, margin_arr_ = 10000,10000,10000,10000
                
                # print(f"Layer {k} | Simulated Capacity (func1)= {sampled_classes_or_exemplars/n_}", flush=True)
                print(f"Layer {k} | Simulated Capacity (func2)= {asim0}", flush=True)
                
                if cfg.WANDB.USE_WANDB:
                    # wandb.log({"mftma_sim_cap_func1": sampled_classes_or_exemplars/n_})
                    wandb.log({"mftma_sim_cap_func2": asim0})

                lst_of_repeated_asim.append(asim0)
                lst_of_repeated_P.append(P_)
                lst_of_repeated_Nc0.append(Nc0)
                lst_of_repeated_N_vec.append(N_vec)
                lst_of_repeated_p_vec.append(p_vec)
                # lst_of_repeated_n_.append(n_)

                lst_of_asim.append(lst_of_repeated_asim)
                lst_of_P.append(lst_of_repeated_P)
                lst_of_Nc0.append(lst_of_repeated_Nc0)
                lst_of_N_vec.append(lst_of_repeated_N_vec)
                lst_of_p_vec.append(lst_of_repeated_p_vec)

                # lst_of_n_.append(lst_of_repeated_n_)
            else:
                # sim_cap_func1 = 0
                asim0 = 0
                print(f"Layer {k} | Simulated Capacity (func2) (skip this calculation)= {asim0}", flush=True)
                if cfg.WANDB.USE_WANDB:
                    # wandb.log({"mftma_sim_cap_func1": sim_cap_func1})
                    wandb.log({"mftma_sim_cap_func2": asim0})

        
        # *************************************************************************** #
        # TODO tmp changes to prevent numerical instability during the R_M calculations

        lst_of_repeated_a = []
        lst_of_repeated_r = []
        lst_of_repeated_d = []
        lst_of_repeated_c = []
        lst_of_repeated_pr = []
        lst_of_repeated_ev = []
        lst_of_repeated_feature = []

        num_recalculation = 0
        while num_recalculation < mftma_num_repeat_calc:
            # , use_sparse_capacity=False, num_of_manifolds=10):
            try:
                a, r, d, r0, K = manifold_analysis_corr(X, 0, 500, n_reps=10, use_sparse_capacity=cfg.MFTMA_ANALYSIS.USE_SPARSE_CAPACITY, num_of_manifolds=sampled_classes_or_exemplars)
            except Exception as e1:
                print(f"yilun | e ={e1}")
                traceback.print_exc()
                try:
                    a, r, d, r0, K = manifold_analysis_corr(X, 0, 500, n_reps=10, use_sparse_capacity=cfg.MFTMA_ANALYSIS.USE_SPARSE_CAPACITY, num_of_manifolds=sampled_classes_or_exemplars)
                except Exception as e2:
                    print(f"yilun | e ={e2}")
                    traceback.print_exc()
                    try:
                        a = lst_of_repeated_a[-1]
                        r = lst_of_repeated_r[-1]
                        d = lst_of_repeated_d[-1] 
                        r0 = lst_of_repeated_c[-1]
                    except Exception as e3:
                        print(f"yilun | e ={e3}")
                        traceback.print_exc()
                        a = 10000
                        r = 10000
                        d = 10000
                        r0 = 10000

            # Compute participation ratio and explained variances
            D_participation_ratio, D_explained_variance, D_feature = alldata_dimension_analysis(X, perc=0.90)

            # Compute the mean values
            a = 1/np.mean(1/a)
            r = np.mean(r)
            d = np.mean(d)

            # Record repeated values
            lst_of_repeated_a.append(a)
            lst_of_repeated_r.append(r)
            lst_of_repeated_d.append(d)
            lst_of_repeated_c.append(r0)
            lst_of_repeated_pr.append(D_participation_ratio)
            lst_of_repeated_ev.append(D_explained_variance)
            lst_of_repeated_feature.append(D_feature)

            print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}, D_participation_ratio {:4f}, D_explained_variance {:4f}, D_feature {:4f}".format(k, a, r, d, r0, D_participation_ratio, D_explained_variance, D_feature))
            
            if cfg.WANDB.USE_WANDB:
                wandb.log({"mftma_layer": k, "mftma_capacity": a, "mftma_radius": r, "mftma_dimension": d, "mftma_correlation": r0, "mftma_D_participation_ratio": D_participation_ratio, "mftma_D_explained_variance": D_explained_variance, "mftma_D_feature": D_feature})
            
            num_recalculation += 1

            if r > 1.5:
                warnings.warn(f"r is too large -> r = {r} @ layer: {k}")

            # Record repeated values
            capacities.append(lst_of_repeated_a)
            radii.append(lst_of_repeated_r)
            dimensions.append(lst_of_repeated_d)
            correlations.append(lst_of_repeated_c)
            lst_D_participation_ratio.append(lst_of_repeated_pr)
            lst_D_explained_variance.append(lst_of_repeated_ev)
            lst_D_feature.append(lst_of_repeated_feature)

            # *************************************************************************** #
    
    if cfg.MFTMA_ANALYSIS.USE_SIM_CAPACITY:
        dict_of_manifold_measure = {
                "asim":lst_of_asim,
                "P":lst_of_P,
                "Nc0":lst_of_Nc0,
                "N_vec":lst_of_N_vec,
                "p_vec":lst_of_p_vec,
                "n_":lst_of_n_,
                "sampled_classes_or_exemplars":sampled_classes_or_exemplars,
            }
    else:
        dict_of_manifold_measure = {
                "capacities":capacities,
                "radii":radii,
                "dimensions":dimensions,
                "correlations":correlations,
                "lst_D_participation_ratio":lst_D_participation_ratio,
                "lst_D_explained_variance":lst_D_explained_variance,
                "lst_D_feature":lst_D_feature,
            }
    
    print(dict_of_manifold_measure)

    save_path = cfg.MFTMA_ANALYSIS.SAVE_PATH
    
    print("done/n")

