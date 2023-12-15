#!/bin/bash
# SSL Objective
# * Available Options: 
# * ssl_training -> ["SimCLR", "MMCR"]
SSL_OBJECTIVE_FUNC=("SimCLR")

#EXPERIMENT NOTES 
#2 augs, 0.01, 512 has decent test acc ~34-35and goodf rfs (100 epochs), good order

# Random Seed
# * Available Options: [42, 43, 44]
SEED=(42) # 43 44)

# SSL Data
# * Available Options: ["real_retinal_waves_three_channels_large", "model_retinal_waves_three_channels_large"]
# SSL_DATASET_NAME=("real_retinal_waves_three_channels_large" "model_retinal_waves_three_channels_large")
# SSL_DATASET_NAME=("real_retinal_waves_three_channels_large" "model_retinal_waves_three_channels_large")
# SSL_DATASET_NAME=("model_retinal_waves_three_channels_large")
SSL_DATASET_NAME=("real_retinal_waves_three_channels_large")
TRAIN_PROP=0.9 #train prop=0.8 for model, 0.9 for real 
BATCH_SIZE_LIMIT=3000
#FOR MFTMA, set pretrained path to None
RAND_PATH=None


# SSL Training Config 
SSL_TRAIN_LEARN_CONTRASTIVE_WAVES=True
LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS=True
FRAME_SEPARATION=(1)

# * Available Options: ["unshuffled", "temporal_shuffling", "spatial_shuffling", "spatial_temporal_shuffling"]
SHUFFLE_TYPE=("unshuffled") # "temporal_shuffling" "spatial_shuffling" "spatial_temporal_shuffling")
# SHUFFLE_TYPE=("spatial_shuffling" "temporal_shuffling" "spatial_temporal_shuffling")
# SHUFFLE_TYPE=("spatial_shuffling")
# SHUFFLE_TYPE=("unshuffled")

TOTAL_EPOCH=100
# TOTAL_EPOCH=500 #yilun
SAVE_EVERY=25

# Classifier Training: CIFAR10 or CIFAR100_classify_augs
# CLASSIFIER_DATASET="CIFAR10"
CLASSIFIER_DATASET="CIFAR100_classify_augs"
#OPTIONS FOR EXEMPLAR TYPE: (translation, rotation, color)
# AUG_TYPE="color"
AUG_TYPE="translation"

CLASSIFIER_TOTAL_EPOCH=5

# MFTMA Config
MFTMA_SAMPLED_CLASSES_OR_EXEMPLARS=50 #10 for cifar10
MFTMA_EXAMPLES_PER_CLASS_OR_EXEMPLARS=20
# * Available Options: ["CIFAR10", "CIFAR100"]
MFTMA_IMAGE_DATASET="CIFAR100"
# * Available Options: ["class", "exemplar"]
MFTMA_IMAGE_MANIFOLD_TYPE="exemplar"
# * Available Options: [True, False]
MFTMA_USE_SPARSE_CAPACITY=False
# * Available Options: [True, False]
MFTMA_USE_SIM_CAPACITY=True
# * Available Options: [True, False]
MFTMA_TEST_WAVES_CAPACITY=True


# Weights and Biases 
# * make sure you have run "wandb.login()" already
WANDB_USE_WANDB=True
# * Available Options: ["retinal_waves_learning", "retinal_waves_learning_SSL", "retinal_waves_learning_Classifier", "retinal_waves_learning_MFTMA"]
WANDB_PROJECT="retinal_waves_learning"


# TODO: Optimizer
# * Available Options: (0.1 0.05 0.01 0.005 0.001 0.0005 0.0001)
# OPTIMIZER_BASE_LR=(0.1 0.05 0.01 0.005 0.001 0.0005 0.0001)
OPTIMIZER_BASE_LR=(0.0001)
# OPTIMIZER_BASE_LR=(0.01)
# OPTIMIZER_BASE_LR=(0.01)
# OPTIMIZER_BASE_LR=(0.0005) 


# TODO: Model
# * Available Options: ["resnet9", "resnet18", "resnet50"]
# MODEL_ENCODER="resnet18"
MODEL_ENCODER="resnet9"
# * Available Options: ("512-128" "2048-512" "4096-1024" "8192-8192-8192")
# MODEL_PROJECTOR_DIMS=("8192-8192-8192")
# MODEL_PROJECTOR_DIMS=("2048-512" "4096-1024" "8192-8192-8192")
# MODEL_PROJECTOR_DIMS=("512-128" "2048-512" "4096-1024" "8192-8192-8192")
# MODEL_PROJECTOR_DIMS=("512-128") #optimal for real
MODEL_PROJECTOR_DIMS=("8192-8192-8192") #optimal for model
# MODEL_PROJECTOR_DIMS=("2048-512" "4096-1024")


# Classifier Sweep
# CLASSIFIER_OPTIMIZER_BASE_LR=(0.0001 0.0005 0.001 0.005) #
# CLASSIFIER_OPTIMIZER_BASE_LR=(0.01 0.05 0.1)
CLASSIFIER_OPTIMIZER_BASE_LR=(0.0001)
# CLASSIFIER_OPTIMIZER_BASE_LR=(0.1 0.05 0.01 0.005 0.001 0.0005 0.0001)
# _CMODELBIAS_LAST=False #overwrites default config



# **************************************************************************************************************************************************************************** #
# *************************************************************************** Config Preprocessing *************************************************************************** #
# **************************************************************************************************************************************************************************** #
if [ $MFTMA_TEST_WAVES_CAPACITY == "True" ]
then
    MANIFOLD_STRING="wave_manifold"
else
    MANIFOLD_STRING="${MFTMA_IMAGE_DATASET}_${MFTMA_IMAGE_MANIFOLD_TYPE}_manifold"
fi

if [ $MFTMA_USE_SIM_CAPACITY == "True" ]
then
    if [ $MFTMA_USE_SPARSE_CAPACITY == "True" ]
    then
        MFTMA_STRING="theo_sim_sparse_mftma_${MFTMA_SAMPLED_CLASSES_OR_EXEMPLARS}manifolds_${MFTMA_EXAMPLES_PER_CLASS_OR_EXEMPLARS}points"
    else
        MFTMA_STRING="theo_sim_dense_mftma_${MFTMA_SAMPLED_CLASSES_OR_EXEMPLARS}manifolds_${MFTMA_EXAMPLES_PER_CLASS_OR_EXEMPLARS}points"
    fi
else
    if [ $MFTMA_USE_SPARSE_CAPACITY == "True" ]
    then
        MFTMA_STRING="theo_sparse_mftma_${MFTMA_SAMPLED_CLASSES_OR_EXEMPLARS}manifolds_${MFTMA_EXAMPLES_PER_CLASS_OR_EXEMPLARS}points"
    else
        MFTMA_STRING="theo_dense_mftma_${MFTMA_SAMPLED_CLASSES_OR_EXEMPLARS}manifolds_${MFTMA_EXAMPLES_PER_CLASS_OR_EXEMPLARS}points"
    fi
fi

if [ $SSL_TRAIN_LEARN_CONTRASTIVE_WAVES == "True" ]
then
    CONTRASTIVE_TYPE="wave_contrastive"
else
    CONTRASTIVE_TYPE="aug_contrastive"
fi

if [ $LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS == "True" ]
then
    COMPRESSION_WINDOW="chop"
else
    COMPRESSION_WINDOW="full"
fi
# **************************************************************************************************************************************************************************** #
# *************************************************************************** Config Preprocessing *************************************************************************** #
# **************************************************************************************************************************************************************************** #



for ssl_objective_func in "${SSL_OBJECTIVE_FUNC[@]}"; do
  for seed in "${SEED[@]}"; do
    for ssl_dataset_name in "${SSL_DATASET_NAME[@]}"; do
      for shuffle_type in "${SHUFFLE_TYPE[@]}"; do
        for ssl_lr in "${OPTIMIZER_BASE_LR[@]}"; do
          for proj_dim in "${MODEL_PROJECTOR_DIMS[@]}"; do
            for frame_sep in "${FRAME_SEPARATION[@]}"; do
              # ! ########################################################################################################################################################## ! #
              # ! ###################################################################### SSL Training ###################################################################### ! #
              # ! ########################################################################################################################################################## ! #
              # WANDB_PROJECT="retinal_waves_learning_SSL"

              # CONFIG_LIST_SSL="OBJECTIVE.OBJECTIVE_TYPE ssl_training \
              #             OBJECTIVE.OBJECTIVE_FUNC $ssl_objective_func \
              #             SEED $seed \
              #             DATA.DATASET_NAME $ssl_dataset_name \
              #             DATA.TRAIN_PROP $TRAIN_PROP \
              #             DATA.BATCH_SIZE_LIMIT $BATCH_SIZE_LIMIT \
              #             SSL_TRAIN.LEARN_CONTRASTIVE_WAVES $SSL_TRAIN_LEARN_CONTRASTIVE_WAVES \
              #             SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS $LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS \
              #             SSL_TRAIN.FRAME_SEPARATION $frame_sep \
              #             SSL_TRAIN.SHUFFLE_TYPE $shuffle_type \
              #             SSL_TRAIN.TOTAL_EPOCH $TOTAL_EPOCH \
              #             OPTIMIZER.BASE_LR $ssl_lr \
              #             MODEL.ENCODER $MODEL_ENCODER \
              #             MODEL.PROJECTOR_DIMS $proj_dim \
              #             MODEL.MODEL_SAVE_PATH /mnt/home/${USER}/ceph/retinal_waves_learning/results/${ssl_dataset_name}/${ssl_objective_func} \
              #             MODEL.MODEL_SAVE_FILE_NAME ${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_${CONTRASTIVE_TYPE}_${COMPRESSION_WINDOW}_${frame_sep}frame_${shuffle_type}_lr${ssl_lr}_projdim${proj_dim}.pt \
              #             WANDB.USE_WANDB $WANDB_USE_WANDB \
              #             WANDB.PROJECT $WANDB_PROJECT \
              #             WANDB.NAME ${ssl_objective_func}_${seed}_${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_${CONTRASTIVE_TYPE}_${COMPRESSION_WINDOW}_${frame_sep}frame_${shuffle_type}_lr${ssl_lr}_projdim${proj_dim} \
              #             SSL_TRAIN.SAVE_EVERY $SAVE_EVERY \
              #             " 

              # echo $CONFIG_LIST_SSL

              # echo -e "\n"
              # cd "/mnt/home/${USER}/ceph/retinal_waves_learning"
              # export PYTHONPATH="/mnt/home/${USER}/ceph/retinal_waves_learning" #sets environment variable for script to find path
              
              # sbatch --job-name=retinal_waves \
              #         --mem=128G \
              #         --gpus=1 \
              #         --gpus-per-node=1 \
              #         --gpus-per-task=1 \
              #         --ntasks-per-node=1 \
              #         --cpus-per-task=10 \
              #         --nodes=1 \
              #         --time=4096 \
              #         --partition=gpu \
              #         --constraint=a100-80gb \
              #         --error=/mnt/home/${USER}/ceph/slurm/retinal_waves/train/%j_%a_%N.err \
              #         --output=/mnt/home/${USER}/ceph/slurm/retinal_waves/train/%j_%a_%N.out \
              #         --export=ALL \
              #         --wrap="python3 scripts/run_experiments.py --cfg_lst \"$CONFIG_LIST_SSL\""
              # ! ########################################################################################################################################################## ! #
              # ! ###################################################################### SSL Training ###################################################################### ! #
              # ! ########################################################################################################################################################## ! #



              # ! ########################################################################################################################################################## ! #
              # ! ################################################################## Classifiers Training ################################################################## ! #
              # ! ########################################################################################################################################################## ! #

              WANDB_PROJECT="retinal_waves_learning_Classifier"
              for classifier_lr in "${CLASSIFIER_OPTIMIZER_BASE_LR[@]}"; do
                # # ? ################################################ ? #
                # # ? Classifier Training - Pretrained Model - CIFAR10 ? #
                # # ? ################################################ ? #
                # CONFIG_LIST_CLASSIFIER="OBJECTIVE.OBJECTIVE_TYPE classifier_training \
                #             OBJECTIVE.OBJECTIVE_FUNC CrossEntropy \
                #             SEED $seed \
                #             DATA.DATASET_NAME $CLASSIFIER_DATASET \
                #             SSL_TRAIN.LEARN_CONTRASTIVE_WAVES $SSL_TRAIN_LEARN_CONTRASTIVE_WAVES \
                #             SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS $LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS \
                #             SSL_TRAIN.FRAME_SEPARATION $frame_sep \
                #             SSL_TRAIN.SHUFFLE_TYPE $shuffle_type \
                #             SSL_TRAIN.TOTAL_EPOCH $TOTAL_EPOCH \
                #             CLASSIFIER_TRAIN.PRETRAINED_SSL_OBJECTIVE $ssl_objective_func \
                #             CLASSIFIER_TRAIN.PRETRAINED_DATASET_NAME $ssl_dataset_name \
                #             CLASSIFIER_TRAIN.TOTAL_EPOCH $CLASSIFIER_TOTAL_EPOCH \
                #             CLASSIFIER_TRAIN.SSL_LR $ssl_lr \
                #             CLASSIFIER_TRAIN.PRETRAINED_PATH /mnt/home/${USER}/ceph/retinal_waves_learning/results/${ssl_dataset_name}/${ssl_objective_func}/${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_${CONTRASTIVE_TYPE}_${COMPRESSION_WINDOW}_${frame_sep}frame_${shuffle_type}_lr${ssl_lr}_projdim${proj_dim}.pt \
                #             CIFAR100_CLASSIFY_AUGS.AUG_TYPE $AUG_TYPE \
                #             OPTIMIZER.BASE_LR $classifier_lr \
                #             MODEL.ENCODER $MODEL_ENCODER \
                #             MODEL.PROJECTOR_DIMS $proj_dim \
                #             MODEL.MODEL_SAVE_PATH /mnt/home/${USER}/ceph/retinal_waves_learning/results/${CLASSIFIER_DATASET}/${ssl_objective_func}/classifier_${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_${CONTRASTIVE_TYPE}_${COMPRESSION_WINDOW}_${frame_sep}frame_${shuffle_type}_ssl_lr${ssl_lr}_ssl_projdim${proj_dim}_classifier_lr_${classifier_lr}.pt \
                #             WANDB.USE_WANDB $WANDB_USE_WANDB \
                #             WANDB.PROJECT $WANDB_PROJECT \
                #             WANDB.NAME ${CLASSIFIER_DATASET}_${ssl_objective_func}_${seed}_classifier_${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_${CONTRASTIVE_TYPE}_${COMPRESSION_WINDOW}_${frame_sep}frame_${shuffle_type}_ssl_lr${ssl_lr}_ssl_projdim${proj_dim}_classifier_lr_${classifier_lr} \
                #             "
              #   # ? ################################################ ? #
              #   # ? Classifier Training - Pretrained Model - CIFAR10 ? #
              #   # ? ################################################ ? #



              #   # ? ################################################ ? #
              #   # ? Classifier Training - Rand Init - CIFAR10 ? #
              #   # ? ################################################ ? #
                CONFIG_LIST_CLASSIFIER="OBJECTIVE.OBJECTIVE_TYPE classifier_training \
                            OBJECTIVE.OBJECTIVE_FUNC CrossEntropy \
                            SEED $seed \
                            DATA.DATASET_NAME $CLASSIFIER_DATASET \
                            SSL_TRAIN.LEARN_CONTRASTIVE_WAVES $SSL_TRAIN_LEARN_CONTRASTIVE_WAVES \
                            SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS $LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS \
                            SSL_TRAIN.FRAME_SEPARATION $frame_sep \
                            SSL_TRAIN.SHUFFLE_TYPE $shuffle_type \
                            SSL_TRAIN.TOTAL_EPOCH $TOTAL_EPOCH \
                            CLASSIFIER_TRAIN.PRETRAINED_SSL_OBJECTIVE $ssl_objective_func \
                            CLASSIFIER_TRAIN.PRETRAINED_DATASET_NAME $ssl_dataset_name \
                            CLASSIFIER_TRAIN.TOTAL_EPOCH $CLASSIFIER_TOTAL_EPOCH \
                            CLASSIFIER_TRAIN.SSL_LR $ssl_lr \
                            CLASSIFIER_TRAIN.PRETRAINED_PATH None \
                            CIFAR100_CLASSIFY_AUGS.AUG_TYPE $AUG_TYPE \
                            OPTIMIZER.BASE_LR $classifier_lr \
                            MODEL.ENCODER $MODEL_ENCODER \
                            MODEL.PROJECTOR_DIMS $proj_dim \
                            MODEL.MODEL_SAVE_PATH /mnt/home/${USER}/ceph/retinal_waves_learning/results/${CLASSIFIER_DATASET}/classifier_${MODEL_ENCODER}_rand_init_seed${seed}_lr${classifier_lr}.pt \
                            WANDB.USE_WANDB $WANDB_USE_WANDB \
                            WANDB.PROJECT $WANDB_PROJECT \
                            WANDB.NAME ${CLASSIFIER_DATASET}_${seed}_classifier_${MODEL_ENCODER}_rand_init_seed${seed}_lr${classifier_lr} \
                            "
              # #   # # ? ################################################ ? #
              # #   # # ? Classifier Training - Rand Init - CIFAR10 ? #
              # #   # # ? ################################################ ? #

                echo $CONFIG_LIST_CLASSIFIER

                echo -e "\n"
                cd "/mnt/home/${USER}/ceph/retinal_waves_learning"
                export PYTHONPATH="/mnt/home/${USER}/ceph/retinal_waves_learning"
              
                sbatch --job-name=retinal_waves \
                      --mem=128G \
                      --gpus=1 \
                      --gpus-per-node=1 \
                      --gpus-per-task=1 \
                      --ntasks-per-node=1 \
                      --cpus-per-task=10 \
                      --nodes=1 \
                      --time=4096 \
                      --partition=gpu \
                      --constraint=a100 \
                      --error=/mnt/home/${USER}/ceph/slurm/retinal_waves/train/%j_%a_%N.err \
                      --output=/mnt/home/${USER}/ceph/slurm/retinal_waves/train/%j_%a_%N.out \
                      --export=ALL \
                      --wrap="python3 scripts/run_experiments.py --cfg_lst \"$CONFIG_LIST_CLASSIFIER\""

              done
              # ! ########################################################################################################################################################## ! #
              # ! ################################################################## Classifiers Training ################################################################## ! #
              # ! ########################################################################################################################################################## ! #





              # ! ########################################################################################################################################################## ! #
              # ! ##################################################################### MFTMA Analysis ##################################################################### ! #
              # ! ########################################################################################################################################################## ! #
              # WANDB_PROJECT="ARCHIVED_resnet_18_MFTMA"
              # WANDB_PROJECT="retinal_waves_learning_MFTMA"

              # ? ##################################################################### ? #
              # ? MFTMA Analysis - Pretrained Model - CIFAR100 & Wave Manifold Capacity ? #
              # ? ##################################################################### ? #
              # CONFIG_LIST_MFTMA="OBJECTIVE.OBJECTIVE_TYPE geometry_analysis \
              #             OBJECTIVE.OBJECTIVE_FUNC MFTMA \
              #             SEED $seed \
              #             DATA.DATASET_NAME $MFTMA_IMAGE_DATASET \
              #             SSL_TRAIN.LEARN_CONTRASTIVE_WAVES $SSL_TRAIN_LEARN_CONTRASTIVE_WAVES \
              #             SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS $LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS \
              #             SSL_TRAIN.SHUFFLE_TYPE $shuffle_type \
              #             SSL_TRAIN.TOTAL_EPOCH $TOTAL_EPOCH \
              #             SSL_TRAIN.FRAME_SEPARATION $frame_sep \
              #             CLASSIFIER_TRAIN.PRETRAINED_SSL_OBJECTIVE $ssl_objective_func \
              #             CLASSIFIER_TRAIN.PRETRAINED_DATASET_NAME $ssl_dataset_name \
              #             CLASSIFIER_TRAIN.SSL_LR $ssl_lr \
              #             OPTIMIZER.BASE_LR $ssl_lr \
              #             MODEL.ENCODER $MODEL_ENCODER \
              #             MODEL.PROJECTOR_DIMS $proj_dim \
              #             MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME $ssl_dataset_name \
              #             MFTMA_ANALYSIS.PRETRAINED_PATH /mnt/home/${USER}/ceph/retinal_waves_learning/results/$ssl_dataset_name/$ssl_objective_func/seed_$seed/${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_wave_contrastive_chop_${frame_sep}frame_${shuffle_type}_lr${ssl_lr}_projdim${proj_dim}.pt \
              #             MFTMA_ANALYSIS.SAVE_PATH /mnt/home/${USER}/ceph/retinal_waves_learning/results/$ssl_dataset_name/$ssl_objective_func/seed_$seed/${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_wave_contrastive_chop_${shuffle_type}_lr${ssl_lr}_projdim${proj_dim}.pt \
              #             MFTMA_ANALYSIS.SAMPLED_CLASSES_OR_EXEMPLARS $MFTMA_SAMPLED_CLASSES_OR_EXEMPLARS \
              #             MFTMA_ANALYSIS.EXAMPLES_PER_CLASS_OR_EXEMPLARS $MFTMA_EXAMPLES_PER_CLASS_OR_EXEMPLARS \
              #             MFTMA_ANALYSIS.USE_SPARSE_CAPACITY $MFTMA_USE_SPARSE_CAPACITY \
              #             MFTMA_ANALYSIS.USE_SIM_CAPACITY $MFTMA_USE_SIM_CAPACITY \
              #             MFTMA_ANALYSIS.WAVES_CAPACITY.TEST_WAVES_CAPACITY $MFTMA_TEST_WAVES_CAPACITY \
              #             MFTMA_ANALYSIS.IMAGE_MANIFOLD_TYPE $MFTMA_IMAGE_MANIFOLD_TYPE \
              #             MFTMA_ANALYSIS.AUG_TYPE $AUG_TYPE \
              #             WANDB.USE_WANDB $WANDB_USE_WANDB \
              #             WANDB.PROJECT $WANDB_PROJECT \
              #             WANDB.NAME ${MFTMA_STRING}_${MANIFOLD_STRING}_${ssl_objective_func}_seed_${seed}_${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_wave_contrastive_chop_${frame_sep}frame_${shuffle_type}_lr${ssl_lr}_projdim${proj_dim} \
              #             "
              # ? ##################################################################### ? #
              # ? MFTMA Analysis - Pretrained Model - CIFAR100 & Wave Manifold Capacity ? #
              # ? ##################################################################### ? #
              
              # # ? ############################################################## ? #
              # # ? MFTMA Analysis - Rand Init - CIFAR100 & Wave Manifold Capacity ? #
              # # ? ############################################################## ? #
              # CONFIG_LIST_MFTMA="OBJECTIVE.OBJECTIVE_TYPE geometry_analysis \
              #             OBJECTIVE.OBJECTIVE_FUNC MFTMA \
              #             SEED $seed \
              #             DATA.DATASET_NAME $MFTMA_IMAGE_DATASET \
              #             SSL_TRAIN.LEARN_CONTRASTIVE_WAVES $SSL_TRAIN_LEARN_CONTRASTIVE_WAVES \
              #             SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS $LEARN_CONTRASTIVE_WAVES_CHOP_WAVES_INTO_SMALLER_SEGMENTS \
              #             SSL_TRAIN.SHUFFLE_TYPE $shuffle_type \
              #             SSL_TRAIN.TOTAL_EPOCH $TOTAL_EPOCH \
              #             SSL_TRAIN.FRAME_SEPARATION $frame_sep \
              #             CLASSIFIER_TRAIN.PRETRAINED_SSL_OBJECTIVE $ssl_objective_func \
              #             CLASSIFIER_TRAIN.PRETRAINED_DATASET_NAME $ssl_dataset_name \
              #             CLASSIFIER_TRAIN.SSL_LR $ssl_lr \
              #             OPTIMIZER.BASE_LR $ssl_lr \
              #             MODEL.ENCODER $MODEL_ENCODER \
              #             MODEL.PROJECTOR_DIMS $proj_dim \
              #             MFTMA_ANALYSIS.PRETRAINED_DATASET_NAME $ssl_dataset_name \
              #             MFTMA_ANALYSIS.PRETRAINED_PATH $RAND_PATH \
              #             MFTMA_ANALYSIS.SAVE_PATH /mnt/home/ykuang/ceph/retinal_waves_learning/results/$ssl_dataset_name/$ssl_objective_func/seed_$seed/${MODEL_ENCODER}_${ssl_dataset_name}_${TOTAL_EPOCH}epoch_wave_contrastive_chop_${shuffle_type}_lr${ssl_lr}_projdim${proj_dim}.pt \
              #             MFTMA_ANALYSIS.SAMPLED_CLASSES_OR_EXEMPLARS $MFTMA_SAMPLED_CLASSES_OR_EXEMPLARS \
              #             MFTMA_ANALYSIS.EXAMPLES_PER_CLASS_OR_EXEMPLARS $MFTMA_EXAMPLES_PER_CLASS_OR_EXEMPLARS \
              #             MFTMA_ANALYSIS.USE_SPARSE_CAPACITY $MFTMA_USE_SPARSE_CAPACITY \
              #             MFTMA_ANALYSIS.USE_SIM_CAPACITY $MFTMA_USE_SIM_CAPACITY \
              #             MFTMA_ANALYSIS.WAVES_CAPACITY.TEST_WAVES_CAPACITY $MFTMA_TEST_WAVES_CAPACITY \
              #             MFTMA_ANALYSIS.IMAGE_MANIFOLD_TYPE $MFTMA_IMAGE_MANIFOLD_TYPE \
              #             MFTMA_ANALYSIS.AUG_TYPE $AUG_TYPE \
              #             WANDB.USE_WANDB $WANDB_USE_WANDB \
              #             WANDB.PROJECT $WANDB_PROJECT \
              #             WANDB.NAME ${MFTMA_STRING}_${MANIFOLD_STRING}_rand_init_seed_${seed}_${MODEL_ENCODER}_projdim${proj_dim} \
              #             "
              # # # ? ############################################################## ? #
              # # # ? MFTMA Analysis - Rand Init - CIFAR100 & Wave Manifold Capacity ? #
              # # # ? ############################################################## ? #

              # echo $CONFIG_LIST_SSL
              # echo $CONFIG_LIST_CLASSIFIER
              # echo $CONFIG_LIST_MFTMA
              # echo -e "\n"

              # cd "/mnt/home/${USER}/ceph/retinal_waves_learning"
              # export PYTHONPATH="/mnt/home/${USER}/ceph/retinal_waves_learning"
              
              # sbatch --job-name=retinal_waves \
              #         --mem=128G \
              #         --gpus=1 \
              #         --gpus-per-node=1 \
              #         --gpus-per-task=1 \
              #         --ntasks-per-node=1 \
              #         --cpus-per-task=10 \
              #         --nodes=1 \
              #         --time=4096 \
              #         --partition=gpu \
              #         --constraint=a100 \
              #         --error=/mnt/home/${USER}/ceph/slurm/retinal_waves/train/%j_%a_%N.err \
              #         --output=/mnt/home/${USER}/ceph/slurm/retinal_waves/train/%j_%a_%N.out \
              #         --export=ALL \
              #         --wrap="python3 scripts/run_experiments.py --cfg_lst \"$CONFIG_LIST_MFTMA\""
              # ! ########################################################################################################################################################## ! #
              # ! ##################################################################### MFTMA Analysis ##################################################################### ! #
              # ! ########################################################################################################################################################## ! #
            done
          done
        done
      done
    done
  done
done
