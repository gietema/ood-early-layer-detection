import os
import torch
import numpy as np
from fastprogress import progress_bar
from contextlib import contextmanager


from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.pooling import MaxPool2d

class LayerOutputExtractor():
    def __init__(self, ds_name:str, layers:list):
        self.layers = layers
        self.feature_path = '/mnt/disks/disk'
        self.name = ds_name
        self.detection_errors = {}

    def get_features(self, layer_idx:int, ds_name:str, ds_type:str = 'test') -> torch.Tensor:
        print(f'getting features for {ds_name} {ds_type}..')
        batched_outputs = []
        layer_dir = f'/mnt/disks/disk/{ds_name}_{ds_type}_layer_output/{layer_idx}'
        batched_output_files = os.listdir(layer_dir)
        for batch_output_files in progress_bar(batched_output_files):
            batch_filename = f'{layer_dir}/{batch_output_files}'
            with self.load_feature_file(batch_filename) as batched_output:
                if isinstance(self.layers[layer_idx], (Conv2d, BatchNorm2d, ReLU, MaxPool2d)):
                    batched_output = self.get_mean_channels(batched_output)
                else:
                    batched_output = torch.tensor(batched_output).to('cpu')
                batched_outputs.append(batched_output)
        return torch.cat(batched_outputs, out=torch.Tensor(len(batched_output_files)*len(batched_output), 64))
    
    def get_mean_channels(self, batched_outputs:list) -> torch.Tensor:
        channel_means = []
        for single_output in batched_outputs:
            channel_means.append([channel.mean() for channel in single_output])
        return torch.tensor(channel_means)
    
    @contextmanager
    def load_feature_file(self, feature_file:str):
        features = torch.load(feature_file)
        try:
            yield features
        finally:
            del features