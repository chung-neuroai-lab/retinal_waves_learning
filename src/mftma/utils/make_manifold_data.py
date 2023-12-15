# Author: cory.stephenson@intel.com

'''
Tool for creating manifold data from a pytorch style dataset
'''
import numpy as np
from collections import defaultdict
import torch
import json
import os
import random

# Custom method for the data augmentation manifold analysis
def make_augmentation_manifold_data(dataset, sampled_manifolds, examples_per_manifold, seed=0):
    # Set the seed
    np.random.seed(seed)
    # Storage for samples and labels
    sampled_data = []
    sampled_label = []

    manifold_index = np.random.choice(len(dataset), size=sampled_manifolds, replace=False)

    for i in manifold_index:
        sample, label = dataset[i]
        sampled_data.append(sample)
        sampled_label.append((i,label))

    return sampled_data

def make_wave_manifold_data(dataset_name, dataset, wave_timepoints, cutoff_wave_length=14):
    if dataset_name == "real_retinal_waves_three_channels_large" or dataset_name == "model_retinal_waves_three_channels_large":
        # get list of wave manifold size/length
        lst_wave_length = wave_timepoints[:,1]-wave_timepoints[:,0]
        
        # initialize sampled waves
        sampled_waves = []
        
        for i, curr_wave_length in enumerate(lst_wave_length):
            curr_wave = dataset[wave_timepoints[i,0]:wave_timepoints[i,1]]
            sampled_waves.append(curr_wave)
    # elif dataset_name == "model_retinal_waves_three_channels_large":
    #     # initialize sampled waves
    #     sampled_waves = []
        
    #     cumsum_wave_timepoints = np.cumsum(wave_timepoints)
    #     for i, wave_label in enumerate(wave_timepoints):
    #         if i == 0:
    #             sampled_waves.append(dataset[0:cumsum_wave_timepoints[i]])
    #         else:
    #             sampled_waves.append(dataset[cumsum_wave_timepoints[i-1]:cumsum_wave_timepoints[i]])    
    else:
        raise ValueError

    return sampled_waves


# def make_manifold_data(dataset, sampled_classes, examples_per_class, max_class=None, seed=0):
#     '''
#     Samples manifold data for use in later analysis

#     Args:
#         dataset: PyTorch style dataset, or iterable that contains (input, label) pairs
#         sampled_classes: Number of classes to sample from (must be less than or equal to
#             the number of classes in dataset)
#         examples_per_class: Number of examples per class to draw (there should be at least
#             this many examples per class in the dataset)
#         max_class (optional): Maximum class to sample from. Defaults to sampled_classes if unspecified
#         seed (optional): Random seed used for drawing samples

#     Returns:
#         data: Iterable containing manifold input data
#     '''
#     if max_class is None:
#         max_class = sampled_classes
#     assert sampled_classes <= max_class, 'Not enough classes in the dataset'
#     assert examples_per_class * max_class <= len(dataset), 'Not enough examples per class in dataset'

#     # Set the seed
#     np.random.seed(seed)
#     # Storage for samples
#     sampled_data = defaultdict(list)
#     # Sample the labels
#     sampled_labels = np.random.choice(list(range(max_class)), size=sampled_classes, replace=False)
#     # Shuffle the order to iterate through the dataset
#     idx = [i for i in range(len(dataset))]
#     np.random.shuffle(idx)
#     # Iterate through the dataset until enough samples are drawn
#     for i in idx:
#         sample, label = dataset[i]
#         if label in sampled_labels and len(sampled_data[label]) < examples_per_class:
#             sampled_data[label].append(sample)
#         # Check if enough samples have been drawn
#         complete = True
#         for s in sampled_labels:
#             if len(sampled_data[s]) < examples_per_class:
#                 complete = False
#         if complete:
#             break
#     # Check that enough samples have been found
#     assert complete, 'Could not find enough examples for the sampled classes'
#     # Combine the samples into batches
#     data = []
#     for s, d in sampled_data.items():
#         data.append(torch.stack(d))
#     return data

# # Custom method for the data augmentation manifold analysis
# def make_augmentation_manifold_data(dataset, sampled_manifolds, examples_per_manifold, seed=0):
#     # Set the seed
#     np.random.seed(seed)
#     # Storage for samples and labels
#     sampled_data = []
#     sampled_label = []

#     manifold_index = np.random.choice(len(dataset), size=sampled_manifolds, replace=False)

#     for i in manifold_index:
#         sample, label = dataset[i]
#         sampled_data.append(sample)
#         sampled_label.append((i,label))

#     return sampled_data

def make_manifold_data(dataset, sampled_classes, examples_per_class, max_class=None, seed=0, dataset_name="CIFAR10",manifold_type="class"):
    '''
    Samples manifold data for use in later analysis

    Args:
        dataset: PyTorch style dataset, or iterable that contains (input, label) pairs
        sampled_classes: Number of classes to sample from (must be less than or equal to
            the number of classes in dataset)
        examples_per_class: Number of examples per class to draw (there should be at least
            this many examples per class in the dataset)
        max_class (optional): Maximum class to sample from. Defaults to sampled_classes if unspecified
        seed (optional): Random seed used for drawing samples

    Returns:
        data: Iterable containing manifold input data
    '''
    if max_class is None:
        max_class = sampled_classes
    assert sampled_classes <= max_class, 'Not enough classes in the dataset'
    assert examples_per_class * max_class <= len(dataset), 'Not enough examples per class in dataset'
    
    user = os.getenv("USER")

    if dataset_name == "CIFAR10":
        if manifold_type == "class":
            with open(f"/mnt/ceph/users/{user}/retinal_waves_learning/src/ranked_instances_cifar10_simclr.json",'r') as f_ind:
                cifar_top_prob_ind = json.load(f_ind)
            print("new cifar10 top prob indices used!", flush=True)
            
            # Set the seed
            # np.random.seed(seed)
            # Storage for samples
            sampled_data = defaultdict(list)
            # Sample the labels
            sampled_labels = np.array(list(range(max_class)))
            # sampled_labels = np.random.choice(list(range(max_class)), size=sampled_classes, replace=False)
            # Shuffle the order to iterate through the dataset
            idx = [i for i in range(len(dataset))]
            # np.random.shuffle(idx)

            # Iterate through the dataset until enough samples are drawn
            for i in idx:
                sample, label = dataset[i]

                if i in cifar_top_prob_ind[str(label)]:
                    sampled_data[label].append(sample)

            # Check that enough samples have been found
            # assert complete, 'Could not find enough examples for the sampled classes'
            # Combine the samples into batches
            data = []
            for s, d in sampled_data.items():
                data.append(torch.stack(d))
        elif manifold_type == "exemplar":
            raise NotImplementedError
        else:
            raise ValueError
    elif dataset_name == "CIFAR100":

        if manifold_type == "class":
            with open(f"/mnt/ceph/users/{user}/retinal_waves_learning/src/ranked_instances_cifar100_simclr.json",'r') as f_ind: #TODO: remove for data aug
                cifar_top_prob_ind = json.load(f_ind)
            print("new cifar100 top prob indices used!", flush=True)
            

            sampled_data = defaultdict(list)
            sampled_labels = random.sample(range(0, 100), max_class)
            idx = [i for i in range(len(dataset))]
            
            for i in idx:
                sample, label = dataset[i]
                if i in cifar_top_prob_ind[str(label)][:examples_per_class] and label in sampled_labels:
                    sampled_data[label].append(sample)

            data = []
            for s, d in sampled_data.items():
                data.append(torch.stack(d))
        elif manifold_type == "exemplar":
            idx = [i for i in range(len(dataset))] 
            sampled_idx = random.sample(range(0, len(idx)), max_class) #sample 50 indices for 50 random images 
            
            data = []
            for i in sampled_idx:
                sample, label = dataset[i] #sample: [n_augs, 3, 32, 32]
                data.append(sample)

        else:
            raise ValueError

    else:
        raise NotImplementedError
    
    return data