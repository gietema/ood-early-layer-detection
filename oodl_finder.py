"""Class to find the Optimal OOD Discernment Layer (OODL)"""
import torch
import os
import numpy as np
from fastprogress import progress_bar
from sklearn.svm import OneClassSVM

class OodlFinder():
    """Find the Optimal OOD Discernment Layer (OODL)"""
    def __init__(self, ds_name:str, layers:list):
        self.layers = layers
        self.feature_path = '/mnt/disks/disk'
        self.name = ds_name
        self.detection_errors = {}
    
    def discover_oodl(self, ood_name:str) -> int:
        """Return the Optimal OOD Discernment Layer"""
        self.calc_all_detection_errors(ood_name)
        return np.array(list(self.detection_errors.values())).argmin()
    
    def calc_all_detection_errors(self, ood_name:str) -> dict:
        """Calculate the detection error for each layer"""
        for layer_idx in range(len(self.layers)):
            self.calc_detection_error_layer(layer_idx, 'tiny_image_net')
        return self.detection_errors
    
    def calc_detection_error_layer(self, layer_idx:int, ood_name:str) -> float:
        """Calculate the detection error of a specific layer"""
        print('\n------------------------------------------')
        print(f'Calculating the detection error for layer {layer_idx}')
        print('------------------------------------------')
        self.model = self.fit_id_features(layer_idx)
        id_features = self.get_features(layer_idx, self.name)
        ood_features = self.get_features(layer_idx, ood_name)
        data = np.vstack((id_features, ood_features))
        print('predict data..')
        preds = self.model.predict(data)
        id_error = 1 - np.count_nonzero(preds[:10000] == 1) / 10000
        ood_error = 1 - np.count_nonzero(preds[10000:] == -1) / 10000
        self.detection_errors[layer_idx] = (id_error + ood_error) / 2
        print(f'Detection error for layer {layer_idx}: {self.detection_errors[layer_idx]}')
        return self.detection_errors[layer_idx]
        
    def fit_id_features(self, layer_idx:int) -> OneClassSVM:
        """Fit a one class SVM on the in-distribution features of a layer"""
        self.in_dist_features = self.get_features(layer_idx, self.name, 'train')
        print('fitting model..')
        model = OneClassSVM(gamma='scale').fit(self.in_dist_features)
        return model
    
    def get_features(self, layer_idx:int, ds_name:str, ds_type:str = 'test') -> torch.Tensor:
        """Get features of layer"""
        print(f'getting features for {ds_name} {ds_type}..')
        batched_outputs = []
        layer_dir = f'/mnt/disks/disk/{ds_name}_{ds_type}_layer_output/{layer_idx}'
        batched_output_files = os.listdir(layer_dir)
        for batch_output_files in progress_bar(batched_output_files):
            with open(f'{layer_dir}/{batch_output_files}', 'rb') as file:
                batched_output = torch.load(file)
                if isinstance(self.layers[layer_idx], torch.nn.modules.conv.Conv2d):
                    batched_output = self.get_mean_channels(batched_output)
                batched_outputs.append(batched_output)
        return torch.cat(batched_outputs, out=torch.Tensor(10000, 64))

    def get_mean_channels(self, batched_outputs:list) -> torch.Tensor:
        """Get mean of each channel of batched features"""
        channel_means = []
        for single_output in batched_outputs:
            channel_means.append([channel.mean() for channel in single_output])
        return torch.tensor(channel_means)
