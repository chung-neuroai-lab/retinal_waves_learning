import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18, resnet101
from src.resnet9 import ResNet9

from torchvision.models.feature_extraction import create_feature_extractor
from torch import Tensor
from typing import Tuple
from typing import OrderedDict

import warnings
from composer.models import ComposerModel
from typing import Any, Tuple


def get_model(
    job_type,

    # model arguments
    encoder: str = "resnet50",
    projector_dims: list = [8192, 8192, 8192],
    pretrain_dataset_name: str = "real_retinal_waves",
    bias_last: bool = False,
    zero_init_residual: bool = False,
    
    # classifier arguments
    pretrained_path: str = None,

    # other optional arguments
    num_class: int = 10,
    load_from_saved_rand_initialization: bool = False,
    cfg = None,
):
    # breakpoint()
    if job_type == "ssl_training":
        model = Model(
            encoder=encoder,
            projector_dims=projector_dims,
            pretrain_dataset_name=pretrain_dataset_name,
            bias_last=bias_last,
            zero_init_residual=zero_init_residual,
            load_from_saved_rand_initialization=load_from_saved_rand_initialization,
            cfg=cfg,
        )
    elif job_type == "classifier_training":
        model = TestModel(
            # classifier
            pretrained_path, 
            cfg,
            # model
            encoder=encoder,
            num_class=num_class,
            projector_dims=projector_dims,
            pretrain_dataset_name=pretrain_dataset_name,
            bias_last=bias_last,
            zero_init_residual=zero_init_residual,
        )

        # For classifier training, only the linear classification layer is trained
        for name, param in model.named_parameters():
            if not 'fc.' in name:
                param.requires_grad = False
            else:
                print(f"classifier_training | requires_grad parameters: {name}.")
    elif job_type == "geometry_analysis":
        model = TestModelMFTMA(
            # classifier
            pretrained_path, 
            
            # model
            encoder=encoder,
            projector_dims=projector_dims,
            pretrain_dataset_name=pretrain_dataset_name,
            bias_last=bias_last,
            zero_init_residual=zero_init_residual,
        )
    else:
        raise NotImplementedError
    
    return model
    
### Linear classifier, for evaluating after unsupervised training has been completed ###
class TestModel(nn.Module):
    def __init__(
        self, 
        # classifier arguments
        pretrained_path,
        cfg, 
        num_class: int = 10,

        # model arguments
        encoder: str = "resnet50",
        projector_dims: list = [8192, 8192, 8192],
        pretrain_dataset_name: str = "real_retinal_waves",
        bias_last: bool = False,
        zero_init_residual: bool = False,
    ):
        super(TestModel, self).__init__()

        self.use_projector = cfg.CLASSIFIER_TRAIN.USE_PROJECTOR
        self.classify_intermediate_layer = cfg.CLASSIFIER_TRAIN.CLASSIFY_INTERMEDIATE_LAYER
        if self.classify_intermediate_layer == True:
            self.classify_layer_num = cfg.CLASSIFIER_TRAIN.CLASSIFY_LAYER_NUM

        if (self.classify_intermediate_layer == True) and (self.use_projector == True):
            raise ValueError
        
        if pretrained_path == None:
            load_rand_init_encoder = True
        else:
            load_rand_init_encoder = False

        if self.use_projector == False:
            self.f = Model(
            encoder=encoder,
            projector_dims=projector_dims,
            pretrain_dataset_name=pretrain_dataset_name,
            bias_last=bias_last,
            zero_init_residual=zero_init_residual,
            load_rand_init_encoder = load_rand_init_encoder,
            ).f
    
        else:
            self.f = Model(
            encoder=encoder,
            projector_dims=projector_dims,
            pretrain_dataset_name=pretrain_dataset_name,
            bias_last=bias_last,
            zero_init_residual=zero_init_residual,
            load_rand_init_encoder = load_rand_init_encoder,
            ).f
            
            self.g = Model(
            encoder=encoder,
            projector_dims=projector_dims,
            pretrain_dataset_name=pretrain_dataset_name,
            bias_last=bias_last,
            zero_init_residual=zero_init_residual,
            load_rand_init_encoder = load_rand_init_encoder,
            ).g
        
        embedding_hidden_size = None
        if encoder == "resnet50":
            if self.use_projector == False:
                embedding_hidden_size = 2048
            else:
                embedding_hidden_size = projector_dims[-1]
        elif encoder == "resnet18":
            if self.use_projector == False:
                embedding_hidden_size = 512
            else:
                embedding_hidden_size = projector_dims[-1]
        elif encoder == "resnet101":
            embedding_hidden_size = 2048
        elif encoder == "resnet9":
            if self.use_projector == False:
                embedding_hidden_size = 1024
            else:
                embedding_hidden_size = projector_dims[-1]
        else:
            raise NotImplementedError

        # assert num_class == 10, "num_class != 10 is not implemented"

        if pretrained_path == None:
            # Perform classifier training over randomly initialized model
            print("Perform classifier training over randomly initialized model")
            pass
        else:
            state_dict = torch.load(pretrained_path, map_location="cpu")

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if ("g." in k) and (cfg.CLASSIFIER_TRAIN.USE_PROJECTOR == False):
                    # skip the projector self.g weights
                    continue
                else:
                    new_state_dict[k] = v

            self.load_state_dict(new_state_dict, strict=True)
            print(f"pretrained_path {pretrained_path} loaded successfully")
        
        if self.classify_intermediate_layer == True:
            node_list = ['relu',
                        'layer1.0.relu',
                        'layer1.1.relu',
                        'layer2.0.relu',
                        'layer2.1.relu',
                        'layer3.0.relu',
                        'layer3.1.relu',
                        'layer4.0.relu',
                        'layer4.1.relu'
                        ]
            self.out_layer = node_list[self.classify_layer_num]
            print(f'Classifying at layer {self.out_layer}')
            self.interm_f = create_feature_extractor(self.f, return_nodes={self.out_layer : 'out'})
            embedding_hidden_size = self.interm_f(torch.rand((1,3,32,32)))['out'].flatten(1).shape[-1]
        # if cfg.DATA.DATASET_NAME == "CIFAR100_classify_augs":
        #     num_class = 50
        self.fc = nn.Linear(embedding_hidden_size, num_class, bias=True)

    def forward(self, x):
        if self.classify_intermediate_layer == True: 
            x = self.interm_f(x)['out']
            out = torch.flatten(x, start_dim=1)
            out = F.normalize(out, dim=-1)
        else:
            x = self.f(x)
            out = torch.flatten(x, start_dim=1)
        if self.use_projector == True:
            out = self.g(out)
            out = F.normalize(out, dim=-1)
        out = self.fc(out)
        return out

### Linear classifier, for evaluating after unsupervised training has been completed ###
class TestModelMFTMA(nn.Module):
    def __init__(
        self, 
        # classifier arguments
        pretrained_path, 
        num_class: int = 10,

        # model arguments
        encoder: str = "resnet50",
        projector_dims: list = [8192, 8192, 8192],
        pretrain_dataset_name: str = "real_retinal_waves",
        bias_last: bool = False,
        zero_init_residual: bool = False,
    ):
        super(TestModelMFTMA, self).__init__()

        embedding_hidden_size = None
        if encoder == "resnet50":
            embedding_hidden_size = 2048
        elif encoder == "resnet18":
            embedding_hidden_size = 512
        elif encoder == "resnet101":
            embedding_hidden_size = 2048
        elif encoder == "resnet9":
            embedding_hidden_size = 1024
        else:
            raise NotImplementedError

        self.f = Model(
            encoder=encoder,
            projector_dims=projector_dims,
            pretrain_dataset_name=pretrain_dataset_name,
            bias_last=bias_last,
            zero_init_residual=zero_init_residual,
        ).f
        self.g = Model(
            encoder=encoder,
            projector_dims=projector_dims,
            pretrain_dataset_name=pretrain_dataset_name,
            bias_last=bias_last,
            zero_init_residual=zero_init_residual,
        ).g

        # self.fc = nn.Linear(embedding_hidden_size, num_class, bias=True) 
        # assert num_class == 10, "num_class != 10 is not implemented"

        if pretrained_path == None:
            # Perform classifier training over randomly initialized model
            print("Perform MFTMA over randomly initialized model")
            pass
        else:
            state_dict = torch.load(pretrained_path, map_location="cpu")

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # if "g." in k:
                #     # skip the projector self.g weights
                #     continue
                # else:
                #     new_state_dict[k] = v
                new_state_dict[k] = v
        
            self.load_state_dict(new_state_dict, strict=True)
            print(f"pretrained_path {pretrained_path} loaded successfully")

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        # out = self.fc(feature)
        return out

class Model(nn.Module):
    def __init__(
        self,
        encoder: str = "resnet50",
        projector_dims: list = [8192, 8192, 8192],
        pretrain_dataset_name: str = "real_retinal_waves",
        bias_last: bool = False,
        zero_init_residual: bool = False,
        load_from_saved_rand_initialization: bool = False,
        cfg = None,
        load_rand_init_encoder = False,
    ):
        super(Model, self).__init__()
        
        embedding_hidden_size = None
        if encoder == "resnet50":
            self.f = resnet50(zero_init_residual=zero_init_residual)
            embedding_hidden_size = 2048
        elif encoder == "resnet18":
            self.f = resnet18(zero_init_residual=zero_init_residual)
            embedding_hidden_size = 512
        elif encoder == "resnet101":
            self.f = resnet101(zero_init_residual=zero_init_residual)
            embedding_hidden_size = 2048
        elif encoder == "resnet9":
            self.f = ResNet9()
            embedding_hidden_size = 1024
        else:
            raise NotImplementedError
        
        if encoder != "resnet9":
            if pretrain_dataset_name == "real_retinal_waves" or pretrain_dataset_name == "model_retinal_waves" or pretrain_dataset_name == "real_retinal_waves_large" or pretrain_dataset_name == "model_retinal_waves_large":
                self.f.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif pretrain_dataset_name == "real_retinal_waves_three_channels" or pretrain_dataset_name == "model_retinal_waves_three_channels" or pretrain_dataset_name == "real_retinal_waves_three_channels_large" or pretrain_dataset_name == "model_retinal_waves_three_channels_large":
                self.f.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
            else:
                raise NotImplementedError

        if encoder != "resnet9":
            if load_rand_init_encoder == True:
                self.f.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
        self.f.fc = nn.Identity()

        # projection head (Following exactly barlow twins offical repo)
        projector_dims = [embedding_hidden_size] + projector_dims
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=False)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.g = nn.Sequential(*layers)

        if load_from_saved_rand_initialization:
            rand_path = os.path.join(cfg.MODEL.MODEL_SAVE_PATH, f"seed_{cfg.SEED}", f"initialization_model_over_{cfg.DATA.DATASET_NAME}_batch_size_100_n_augs_{cfg.DATA.N_AUGS}.pt")
            state_dict = torch.load(rand_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
            print(f"rand_path {rand_path} loaded successfully")

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)

        # normalize (project to unit sphere)
        out = F.normalize(out, dim=-1)

        return feature, out

