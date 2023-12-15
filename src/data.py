import os
import cv2
import glob
import mat73
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from src import preprocessing
from src.layers import Retina

def get_dataset(
    dataset_name,
    dataset_path,
    batch_size,
    n_augs,
    use_shuffle,
    # use_validation,
    num_workers,

    # Real Retinal Waves
    load_saved_retinal_waves: bool = True,
    n_retina_neurons: int = 1024,

    # MNIST
    mnist_train_shuffle: bool = True,
    mnist_test_shuffle: bool = False,

    # CIFAR10
    cifar_train_shuffle: bool = True,
    cifar_test_shuffle: bool = False,

    #!CIFAR100 Augmentation (for classification)
    targets_path = None,
    aug_type = None,

    #!CIFAR100 Augmentation Transform (for manifolds)
    exemplar_images_aug_transform: bool = False,
    n_transform: int = 50,
    
):
    user = os.getenv("USER")
    if dataset_name == "real_retinal_waves" or dataset_name=="real_retinal_waves_three_channels":
        retinal_image_dataset_path = get_real_retinal_waves_dataset(
            dataset_name = dataset_name,
            load_saved_retinal_waves = load_saved_retinal_waves,
            retinal_image_dataset_path = dataset_path,
            n_retina_neurons = n_retina_neurons,
        )
        data = RetinalWavesDataset(retinal_image_dataset_path)
        train_dataloader = DataLoader(
            data, 
            batch_size=batch_size*n_augs,
            shuffle=use_shuffle, 
            num_workers=num_workers,
        )
        test_dataloader = None
    elif dataset_name == "real_retinal_waves_large" or dataset_name=="real_retinal_waves_three_channels_large":
        retinal_image_dataset_path = get_real_retinal_waves_dataset(
            dataset_name = dataset_name,
            load_saved_retinal_waves = load_saved_retinal_waves,
            retinal_image_dataset_path = dataset_path,
            n_retina_neurons = n_retina_neurons,
        )
        #default_args 
        train_data = RetinalWavesDataset(retinal_image_dataset_path)
        train_dataloader = DataLoader(
            train_data, 
            batch_size=batch_size*n_augs,
            shuffle=use_shuffle, 
            num_workers=num_workers,
        )
        # if use_validation == True:
        #     val_data = RetinalWavesDataset(retinal_image_dataset_path, train=False)
        #     val_dataloader = DataLoader(
        #         val_data, 
        #         batch_size=batch_size*n_augs,
        #         shuffle=use_shuffle, 
        #         num_workers=num_workers,
        #     )
        # else:
        #     val_dataloader = None
        test_dataloader = None
    elif dataset_name == "model_retinal_waves" or dataset_name == "model_retinal_waves_three_channels":
        retinal_image_dataset_path = get_model_retinal_waves_dataset(
            dataset_name = dataset_name,
            load_saved_retinal_waves = load_saved_retinal_waves,
            retinal_image_dataset_path = dataset_path,
            n_retina_neurons = n_retina_neurons,
        )
        data = RetinalWavesDataset(retinal_image_dataset_path)
        train_dataloader = DataLoader(
            data, 
            batch_size=batch_size*n_augs,
            shuffle=use_shuffle, 
            num_workers=num_workers,
        )
        test_dataloader = None
    elif dataset_name == "model_retinal_waves_large" or dataset_name == "model_retinal_waves_three_channels_large":
        retinal_image_dataset_path = get_model_retinal_waves_dataset(
            dataset_name = dataset_name,
            load_saved_retinal_waves = load_saved_retinal_waves,
            retinal_image_dataset_path = dataset_path,
            n_retina_neurons = n_retina_neurons,
        )
        data = RetinalWavesDataset(retinal_image_dataset_path)
        train_dataloader = DataLoader(
            data, 
            batch_size=batch_size*n_augs,
            shuffle=use_shuffle, 
            num_workers=num_workers,
        )
        test_dataloader = None
    elif dataset_name == "simulated_retinal_waves":
        raise NotImplementedError
    elif dataset_name == "MNIST":
        train_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": False,
            "shuffle": mnist_train_shuffle,
        }
        test_kwargs = {
            "batch_size": 1000,
            "num_workers": num_workers,
            "pin_memory": False,
            "shuffle": mnist_test_shuffle,
        }
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_set = datasets.MNIST(
            dataset_path, train=True, download=False, transform=test_transform
        )
        test_set = datasets.MNIST(
            dataset_path, train=False, transform=test_transform
        )
        train_dataloader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        test_dataloader = torch.utils.data.DataLoader(test_set, **test_kwargs)
        print(f"train_set={train_set}")
        print(f"test_set={test_set}")
    elif dataset_name == "CIFAR10":
        train_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": False,
            "shuffle": cifar_train_shuffle,
        }
        test_kwargs = {
            "batch_size": 1000,
            "num_workers": num_workers,
            "pin_memory": False,
            "shuffle": cifar_test_shuffle,
        }
        cifar_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
                ),
            ]
        )
        train_set = datasets.CIFAR10(
            dataset_path,
            train=True,
            download=False,
            transform=cifar_transform,
        )
        test_set = datasets.CIFAR10(
            dataset_path, 
            train=False, 
            transform=cifar_transform,
        )
        print(f"train_set={train_set}")
        print(f"test_set={test_set}")
        train_dataloader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        test_dataloader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    #!CLASSIFICATION FOR AUGS
    elif dataset_name == "CIFAR100_classify_augs":
        train_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": False,
            "shuffle": cifar_train_shuffle,
        }
        test_kwargs = {
            "batch_size": 1000,
            "num_workers": num_workers,
            "pin_memory": False,
            "shuffle": cifar_test_shuffle,
        }
        # cifar_transform = transforms.Compose(
        #     [
        #         # transforms.ToTensor(),
        #         transforms.Normalize(
        #             [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        #         ),
        #     ]
        # )
        if aug_type == "translation":
            transform = CIFAR100TranslateTransform(shift_proportion=0.5)
        elif aug_type == "rotation":
            transform = CIFAR100TranslateTransform(angle=180)
        # elif aug_type == "scaling": 
        elif aug_type == "color":
            transform = CIFAR100ColorTransform()

        trainset = CIFAR100AugDataset(targets_path, dataset_path, train=True, transform=transform)
        testset = CIFAR100AugDataset(targets_path, dataset_path, train=False, transform=transform)

        train_dataloader = torch.utils.data.DataLoader(trainset, **train_kwargs)
        
        test_dataloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    #!CLASSIFICATION FOR AUGS ^^^

    elif dataset_name == "CIFAR100": 
        if aug_type == "translation":
            transform = CIFAR100_Translation_Manifold_Transform(n_transform=n_transform, exemplar_images_aug_transform=exemplar_images_aug_transform)
        elif aug_type == "color":
            transform = CIFAR100_Color_Manifold_Transform(n_transform=n_transform, exemplar_images_aug_transform=exemplar_images_aug_transform)
        elif aug_type == "class":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
            
        trainset = datasets.CIFAR100(root=dataset_path, train=True,
                                                download=True, transform=transform)
        testset = datasets.CIFAR100(root=dataset_path, train=False,
                                            download=True, transform=transform)
        # TODO: print out dir(trainset). trainset.data -> get pytorch tensor -> store the selected tensor
        # print out dir(trainset). trainset.data -> get pytorch tensor -> store the selected tensor
        # load 

        # Create data loaders to load the data in batches
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=cifar_train_shuffle, num_workers=num_workers)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                                shuffle=cifar_test_shuffle, num_workers=num_workers)
    else:
        raise NotImplementedError
    # if use_validation == False:
    #     return train_dataloader, test_dataloader
    # else:
    #     return train_dataloder, test_dataloader, val_dataloader
    return train_dataloader, test_dataloader

def get_model_retinal_waves_dataset(
    dataset_name: str = "model_retinal_waves",
    load_saved_retinal_waves: bool = True,
    retinal_image_dataset_path: str = "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/model_wave_tensor.pt",
    n_retina_neurons: int = 1024,
    CHW_DIM: list = [1,32,32],
):
    # Load Real Retinal Waves Dataset
    if load_saved_retinal_waves:
        # torch.Size([72000, 1, 32, 32]) or torch.Size([72000, 3, 32, 32])
        retinal_image_dataset = torch.load(retinal_image_dataset_path) 
        if dataset_name == "model_retinal_waves" or dataset_name == "model_retinal_waves_large":
            assert retinal_image_dataset.shape[1]==1
        elif dataset_name == "model_retinal_waves_three_channels" or dataset_name == "model_retinal_waves_three_channels_large":
            assert retinal_image_dataset.shape[1]==3
        else:
            raise NotImplementedError
    else:
        # Manual override; KEEP THIS PARAM FIXED even if the dataset is "real_retinal_waves_three_channels"
        # CHW_DIM[0] = 1
        if dataset_name == "model_retinal_waves" or dataset_name=="model_retinal_waves_three_channels":
            frames = mat73.loadmat("/mnt/home/aligeralde/ceph/retinal_waves_learning/data/rw_model_72k_V.mat")['V']
            frames = frames[0:72000]
            frames = torch.from_numpy(frames)

            scaler = StandardScaler()
            model_waves = scaler.fit_transform(frames.reshape(72000,32*32))
            model_waves = (model_waves - model_waves.min())/(model_waves.max()-model_waves.min())
            model_waves = model_waves.reshape(72000,32,32)  # shape: (72000, 128, 128)
            model_waves = model_waves.reshape([72000, 1, 32, 32])
            model_waves = torch.from_numpy(model_waves)
        elif dataset_name == "model_retinal_waves_large" or dataset_name=="model_retinal_waves_three_channels_large":
            # newest version
            frames = mat73.loadmat("/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_model_only_waves1.mat")['V']
            frames = frames[0:20000]

            scaler = StandardScaler()
            model_waves = scaler.fit_transform(frames.reshape(20000,32*32))
            model_waves = (model_waves - model_waves.min())/(model_waves.max()-model_waves.min())

            model_waves = model_waves.reshape(20000,32,32)
            model_waves = torch.from_numpy(model_waves)
            model_waves = model_waves.unsqueeze(1)

            # old version
            # model_waves = np.load("/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_model.npy")
            # model_waves = torch.from_numpy(model_waves)

            # scaler = StandardScaler()
            # model_waves = scaler.fit_transform(model_waves.reshape(20000,32*32))
            # model_waves = (model_waves - model_waves.min())/(model_waves.max()-model_waves.min())
            # model_waves = model_waves.reshape(20000,32,32)

            # model_waves = torch.from_numpy(model_waves)
            # model_waves = model_waves.unsqueeze(1)
        else:
            raise NotImplementedError

        # Compute one/three channels real retinal waves dataset
        if dataset_name == "model_retinal_waves" or dataset_name == "model_retinal_waves_large":
            retinal_image_dataset = model_waves
        elif dataset_name == "model_retinal_waves_three_channels" or dataset_name == "model_retinal_waves_three_channels_large":
            retinal_image_dataset = torch.cat([model_waves,model_waves,model_waves],dim=1)
        else:
            raise NotImplementedError

        # Save retinal_imageset
        try:
            torch.save(retinal_image_dataset.float(), retinal_image_dataset_path)
        except Exception as e:
            print(e)
            torch.save(retinal_image_dataset.float(), ".")
    
    return retinal_image_dataset_path

def get_real_retinal_waves_dataset(
    dataset_name: str = "real_retinal_waves",
    load_saved_retinal_waves: bool = True,
    retinal_image_dataset_path: str = "/mnt/home/aligeralde/ceph/retinal_waves_learning/data/real_wave_tensor.pt",
    n_retina_neurons: int = 1024,
    CHW_DIM: list = [1,32,32],
):
    # Load Real Retinal Waves Dataset
    if load_saved_retinal_waves:
        # torch.Size([72000, 1, 32, 32]) or torch.Size([72000, 3, 32, 32])
        retinal_image_dataset = torch.load(retinal_image_dataset_path) 
        if dataset_name == "real_retinal_waves":
            assert retinal_image_dataset.shape[1]==1
        elif dataset_name == "real_retinal_waves_three_channels":
            assert retinal_image_dataset.shape[1]==3
        elif dataset_name == "real_retinal_waves_large":
            assert retinal_image_dataset.shape[1]==1
        elif dataset_name == "real_retinal_waves_three_channels_large":
            assert retinal_image_dataset.shape[1]==3
        else:
            raise NotImplementedError
    else:
        if dataset_name == "real_retinal_waves" or dataset_name=="real_retinal_waves_three_channels":
            # Define Retina
            retina = Retina(n_neurons=n_retina_neurons)

            # Manual override; KEEP THIS PARAM FIXED even if the dataset is "real_retinal_waves_three_channels"
            CHW_DIM[0] = 1

            data1 = io.imread('../data/210830_001-oriented.tif') # shape: (3600, 128, 128)
            data2 = io.imread('../data/210830_002-oriented.tif') # shape: (3600, 128, 128)
            data3 = io.imread('../data/210830_003-oriented.tif') # shape: (3600, 128, 128)
            data4 = io.imread('../data/210830_004-oriented.tif') # shape: (3600, 128, 128)
            data5 = io.imread('../data/210830_005-oriented.tif') # shape: (3600, 128, 128)
            real_waves = np.concatenate((data1, 
                                        data2, 
                                        data3, 
                                        data4, 
                                        data5, 
                                        np.rot90(data1, k=1, axes=(1,2)),
                                        np.rot90(data2, k=1, axes=(1,2)),
                                        np.rot90(data3, k=1, axes=(1,2)),
                                        np.rot90(data4, k=1, axes=(1,2)),
                                        np.rot90(data5, k=1, axes=(1,2)),
                                        np.rot90(data1, k=2, axes=(1,2)),
                                        np.rot90(data2, k=2, axes=(1,2)),
                                        np.rot90(data3, k=2, axes=(1,2)),
                                        np.rot90(data4, k=2, axes=(1,2)),
                                        np.rot90(data5, k=2, axes=(1,2)),
                                        np.rot90(data1, k=3, axes=(1,2)),
                                        np.rot90(data2, k=3, axes=(1,2)),
                                        np.rot90(data3, k=3, axes=(1,2)),
                                        np.rot90(data4, k=3, axes=(1,2)),
                                        np.rot90(data5, k=3, axes=(1,2)),
                                        ), axis=0) # shape: (3600 * 20, 128, 128) = (72000, 128, 128)

            # Scale and normalize data
            scaler = StandardScaler()
            real_waves = scaler.fit_transform(real_waves.reshape(72000,128*128))
            real_waves = (real_waves - real_waves.min())/(real_waves.max()-real_waves.min())
            real_waves = real_waves.reshape(72000,128,128)  # shape: (72000, 128, 128)
            
            # Load data and project onto retinal grid
            real_wave_dataset = preprocessing.RetinalLargeImageDataset(retina, real_waves)
            real_wave_dataset.project_data_onto_retina(binarize=False)
            retinal_image_dataset = real_wave_dataset.retinal_imageset # torch.Size([1024, 72000])
            retinal_image_dataset = retinal_image_dataset.T
            retinal_image_dataset = retinal_image_dataset.reshape([retinal_image_dataset.shape[0],CHW_DIM[0],CHW_DIM[1],CHW_DIM[2]])
        elif dataset_name == "real_retinal_waves_large" or dataset_name=="real_retinal_waves_three_channels_large":
            retinal_image_dataset = np.load("/mnt/home/aligeralde/ceph/retinal_waves_learning/data/large_area_real_data.npy")
            retinal_image_dataset = torch.from_numpy(retinal_image_dataset)

            scaler = StandardScaler()
            retinal_image_dataset = scaler.fit_transform(retinal_image_dataset.reshape(20000,32*32))
            retinal_image_dataset = (retinal_image_dataset - retinal_image_dataset.min())/(retinal_image_dataset.max()-retinal_image_dataset.min())
            retinal_image_dataset = retinal_image_dataset.reshape(20000,32,32)
            retinal_image_dataset = torch.from_numpy(retinal_image_dataset)

            retinal_image_dataset = retinal_image_dataset.unsqueeze(1) # torch.Size([20000, 1, 32, 32])
        else:
            raise NotImplementedError
        
        # TODO fix the logic below
        # Compute one/three channels real retinal waves dataset
        if dataset_name == "real_retinal_waves":
            pass
        elif dataset_name == "real_retinal_waves_large":
            pass
        elif dataset_name == "real_retinal_waves_three_channels":
            retinal_image_dataset = torch.cat([retinal_image_dataset,retinal_image_dataset,retinal_image_dataset],dim=1)
        elif dataset_name == "real_retinal_waves_three_channels_large":
            retinal_image_dataset = torch.cat([retinal_image_dataset,retinal_image_dataset,retinal_image_dataset],dim=1)
        else:
            raise NotImplementedError

        # Save retinal_imageset
        try:
            torch.save(retinal_image_dataset, retinal_image_dataset_path)
        except Exception as e:
            print(e)
            torch.save(retinal_image_dataset, ".")
    
    return retinal_image_dataset_path

def handling_retinal_wave_shuffling(
    use_temporal_shuffling, # cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_TEMPORAL_SHUFFLE
    use_spatial_shuffling, # cfg.SSL_TRAIN.LEARN_CONTRASTIVE_WAVES_SPATIAL_SHUFFLE
    seeded_data_path, # seeded_data_path
    curr_dataset_name, # cfg.DATA.DATASET_NAME
    wave_manifolds_data,
):
    # spatio-temporal shuffling
    if use_temporal_shuffling and use_spatial_shuffling:
        typed_seeded_data_path = os.path.join(seeded_data_path, "spatio_temporal_shuffle")
        shuffling_data_name = os.path.join(typed_seeded_data_path, f"{curr_dataset_name}_spatio_temporal_shuffle.pt")
        if not os.path.exists(typed_seeded_data_path):
            os.mkdir(typed_seeded_data_path)

        random_permute_indices = torch.randperm(wave_manifolds_data.shape[0])
        wave_manifolds_data = wave_manifolds_data[random_permute_indices]
        
        tmp_wave_manifolds_data = wave_manifolds_data.reshape(len(wave_manifolds_data),3,32*32)
        for i in range(tmp_wave_manifolds_data.shape[0]):
            for j in range(tmp_wave_manifolds_data.shape[1]):
                random_permute_indices = torch.randperm(32*32)
                tmp_wave_manifolds_data[i,j,:] = tmp_wave_manifolds_data[i,j,:][random_permute_indices]

        wave_manifolds_data = tmp_wave_manifolds_data.reshape(len(wave_manifolds_data),3,32,32)   
        #torch.save(wave_manifolds_data, shuffling_data_name)
        print("spatio_temporal shuffling complete!")   
    # temporal shuffling
    elif use_temporal_shuffling:
        typed_seeded_data_path = os.path.join(seeded_data_path, "temporal_shuffle")
        shuffling_data_name = os.path.join(typed_seeded_data_path, f"{curr_dataset_name}_temporal_shuffle.pt")
        if not os.path.exists(typed_seeded_data_path):
            os.mkdir(typed_seeded_data_path)

        random_permute_indices = torch.randperm(wave_manifolds_data.shape[0])
        wave_manifolds_data = wave_manifolds_data[random_permute_indices]
        #torch.save(wave_manifolds_data, shuffling_data_name)
        print("temporal shuffling complete!")
    # spatial shuffling
    elif use_spatial_shuffling:
        typed_seeded_data_path = os.path.join(seeded_data_path, "spatial_shuffle")
        shuffling_data_name = os.path.join(typed_seeded_data_path, f"{curr_dataset_name}_spatial_shuffle.pt")
        if not os.path.exists(typed_seeded_data_path):
            os.mkdir(typed_seeded_data_path)

        
        tmp_wave_manifolds_data = wave_manifolds_data.reshape(len(wave_manifolds_data),3,32*32)
        for i in range(tmp_wave_manifolds_data.shape[0]):
            for j in range(tmp_wave_manifolds_data.shape[1]):
                random_permute_indices = torch.randperm(32*32)
                tmp_wave_manifolds_data[i,j,:] = tmp_wave_manifolds_data[i,j,:][random_permute_indices]

        wave_manifolds_data = tmp_wave_manifolds_data.reshape(len(wave_manifolds_data),3,32,32)  
        #torch.save(wave_manifolds_data, shuffling_data_name) 
        print("spatial shuffling complete!")         
    else:
        print("no shuffling!")
    
    return wave_manifolds_data

def generate_retinal_wave_video(
    retinal_wave_file_name: str,

    # saving
    user: str,
    wave_type: str,
    seed: int,
    shuffle_type: str,
):  
    if user != "aligeralde":
        raise NotImplementedError
    else:
        data_base_path = "/mnt/home/aligeralde/ceph/retinal_waves_learning/data"
        assert wave_type in ["real_retinal_waves_three_channels_large", "model_retinal_waves_three_channels_large"]
        assert seed in [42, 43, 44, 45, 46]
        assert shuffle_type in ["spatial_shuffle", "spatio_temporal_shuffle", "temporal_shuffle"]

        # get save_path
        save_path = os.path.join(data_base_path, wave_type, f"seed_{seed}", shuffle_type)

        # create path for images and videos
        if not os.path.exists(os.path.join(save_path, "images")):
            os.mkdir(os.path.join(save_path, "images"))
        if not os.path.exists(os.path.join(save_path, "videos")):
            os.mkdir(os.path.join(save_path, "videos"))

        # load retinal wave dataset
        retinal_wave_path = os.path.join(save_path, retinal_wave_file_name)
        retinal_wave_dataset = torch.load(retinal_wave_path)

        # pick only the first channels
        retinal_wave_dataset = retinal_wave_dataset[:,0:1,:,:].cpu()

        # Create a sequence of PyTorch tensors
        tensors = [retinal_wave_dataset[i].permute(1,2,0) for i in range(len(retinal_wave_dataset))]

        # Convert the tensors to numpy arrays
        arrays = [t.numpy() for t in tensors]

        # Normalize the arrays between 0 and 255
        arrays = [(a * 255 / a.max()).astype(np.uint8) for a in arrays]

        # Write the arrays as images
        for i, array in enumerate(arrays):
            cv2.imwrite(f"{os.path.join(save_path, 'images')}/frame_{i}.jpg", array)
        
        # load the saved images
        img_array = []
        for filename in glob.glob(f"{os.path.join(save_path, 'images')}/*.jpg"):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        
        # write to video
        out = cv2.VideoWriter(f"{os.path.join(save_path, 'videos')}/{wave_type}_{seed}_{shuffle_type}_video.avi",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

class RetinalWavesDataset(Dataset):
    """Retinal Waves dataset."""
    
    def __init__(self, dataset_path, transform=None):
        self.retinal_image_dataset = torch.load(dataset_path) # self.retinal_image_dataset is a pytorch tensor
        self.retinal_image_dataset = self.retinal_image_dataset.float()
        # #for resetting original rng state
        # old_rng_state = torch.random.get_rng_state()

        # #for reproducing data split
        # torch.manual_seed(10)

        # #load whole dataset and get indices for splitting
        # data = torch.load(dataset_path)
        # rand_data_indices = torch.randperm(len(data))
        # data = data[rand_data_indices]

        # #reset rng state
        # torch.random.set_rng_state(old_rng_state)

        # if train == True:
        #     #get train set
        #     self.retinal_image_dataset = data[:train_prop*len(rand_data_indices)]
        #     self.retinal_image_dataset = self.retinal_image_dataset.float()
        
        # else:
        #     #get validation set
        #     self.retinal_image_dataset = data[train_prop*len(rand_data_indices):]
        #     self.retinal_image_dataset = self.retinal_image_dataset.float()

        # TODO: Remove. Probably not necessary for now
        self.transform = transform

    def __len__(self):
        return len(self.retinal_image_dataset)

    def __getitem__(self, idx):
        # one image from cifar100_subset, [1, 3, 32, 32]
        image = self.retinal_image_dataset[idx]

        
        # TODO: remove
        # image = self.transform(image)
        # one image from cifar100_subset, [n_augs, 3, 32, 32]
        return image

#!FOR CLASSIFIER ONLY
class CIFAR100AugDataset(Dataset):
    """Retinal Waves dataset."""
    
    def __init__(self, path_to_targets_tensor, path_to_image_tensor, train=True, transform=None):
        self.images = torch.load(path_to_image_tensor) # self.images is a pytorch tensor 
        self.images = self.images.float()
        self.targets = torch.load(path_to_targets_tensor)

        #training data in first 5/6 of the image tensor
        if train == True:
            self.images = self.images[:len(self.images)*5//6]
            self.targets = self.targets[:len(self.targets)*5//6]
        
        #test data in last 5/6 of the image tensor
        else:
            self.images = self.images[len(self.images)*5//6:]
            self.targets = self.targets[len(self.targets)*5//6:]

        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # one image from cifar100_subset, [1, 3, 32, 32]
        image = self.images[idx]
        image = self.transform(image)
        target = self.targets[idx]
        # one image from cifar100_subset, [n_augs, 3, 32, 32]
        return image, target

#!FOR CLASSIFIER ONLY
class CIFAR100TranslateTransform:
    def __init__(
        self,
        shift_proportion: float = 0,
        angle: float = 0,
        # exemplar_images_aug_transform: bool = False,
    ):
        self.shift_proportion = shift_proportion
        self.angle = angle

        # self.exemplar_images_aug_transform = exemplar_images_aug_transform

        # if self.exemplar_images_aug_transform:
        lst_of_transform = [
            transforms.RandomAffine(self.angle, translate=(self.shift_proportion,self.shift_proportion)),
            # transforms.ToTensor(),
            transforms.Normalize(
                [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            ),
        ]
        self.transform = transforms.Compose(lst_of_transform)

    def __call__(self, x):
        return self.transform(x)

class CIFAR100ColorTransform:
    def __init__(
        self,
        # shift_proportion: float = 0,
        # angle: float = 0,
        # exemplar_images_aug_transform: bool = False,
    ):
        # self.shift_proportion = shift_proportion
        # self.angle = angle

        # self.exemplar_images_aug_transform = exemplar_images_aug_transform

        # if self.exemplar_images_aug_transform:
        lst_of_transform = [
            transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
            # transforms.ToTensor(),
            transforms.Normalize(
                [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            ),
        ]
        self.transform = transforms.Compose(lst_of_transform)

    def __call__(self, x):
        return self.transform(x)


#!FOR MANIFOLD ONLY
class CIFAR100_Translation_Manifold_Transform:
    def __init__(
        self,
        n_transform: int = 50,
        exemplar_images_aug_transform: bool = False,
    ):
        self.n_transform = n_transform
        self.exemplar_images_aug_transform = exemplar_images_aug_transform

        if self.exemplar_images_aug_transform:
            lst_of_transform = [
                # transforms.RandomResizedCrop(32),
                transforms.RandomAffine(0, translate=(.1,.1)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                # ),
                # transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                ),
            ]
            self.transform = transforms.Compose(lst_of_transform)
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                    ),
                ]
            )

    def __call__(self, x):
        if self.exemplar_images_aug_transform:
            C, H, W = TF.to_tensor(x).shape
            C_aug, H_aug, W_aug = self.transform(x).shape

            y = torch.zeros(self.n_transform, C_aug, H_aug, W_aug)
            for i in range(self.n_transform):
                y[i, :, :, :] = self.transform(x)
            return y
        else:
            return self.transform(x)

class CIFAR100_Color_Manifold_Transform:
    def __init__(
        self,
        n_transform: int = 50,
        exemplar_images_aug_transform: bool = False,
    ):
        self.n_transform = n_transform
        self.exemplar_images_aug_transform = exemplar_images_aug_transform

        if self.exemplar_images_aug_transform:
            lst_of_transform = [
                # transforms.RandomResizedCrop(32),
                transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                # ),
                # transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                ),
            ]
            self.transform = transforms.Compose(lst_of_transform)
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                    ),
                ]
            )

    def __call__(self, x):
        if self.exemplar_images_aug_transform:
            C, H, W = TF.to_tensor(x).shape
            C_aug, H_aug, W_aug = self.transform(x).shape

            y = torch.zeros(self.n_transform, C_aug, H_aug, W_aug)
            for i in range(self.n_transform):
                y[i, :, :, :] = self.transform(x)
            return y
        else:
            return self.transform(x)