import umap
import matplotlib.pyplot as plt
import numpy as np
from layer_output_extractor import LayerOutputExtractor

class Umapper():
    def __init__(self, ds_name, layers, layer_idx):
        self.ds_name = ds_name
        self.layers = layers
        self.layer_idx = layer_idx
    
    def fit_umap_for_layer(self, ood_ds_name):
        reducer = umap.UMAP()
        self.ood_ds_name = ood_ds_name
        id_outputs = self.get_layer_output(self.layer_idx, self.ds_name)
        ood_outputs = self.get_layer_output(self.layer_idx, self.ood_ds_name)
        print('fitting umap model..')
        self.model = reducer.fit_transform(np.vstack((id_outputs, ood_outputs)))
    
    def plot_umap(self, save:bool = False, save_dir:str = None):
        plt.figure(figsize=(20,20))
        plt.title(f'UMAP of layer {self.layer_idx}: {type(self.layers[self.layer_idx]).__name__}', fontsize=24)
        plt.scatter(self.model[:10000, 0], self.model[:10000, 1], alpha=.5, label=self.ds_name)
        plt.scatter(self.model[10000:, 0], self.model[10000:, 1], alpha=.5, label=self.ood_ds_name)
        plt.legend(fontsize=24)
        plt.axis('off')
        if save:
            plt.savefig(save_dir+'/'+str(self.layer_idx)+'.png')
        plt.show()
        
    def get_layer_output(self, layer_idx, ds_name:str, ds_type:str = 'test'):
        return LayerOutputExtractor(self.ds_name, self.layers).get_features(layer_idx, ds_name, ds_type)
