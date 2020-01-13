class FeatureInferenceExtractor(HookCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_idx = 0
        self.batch_idx = 0
        self.stats = []
        self.n_epoch = 1
        
    def hook(self, m:nn.Module, i:Tensors, o:Tensors)-> Tensors:
        return o

    def on_batch_end(self, train, *args, **kwargs):
        for layer_idx, o in enumerate(self.hooks.stored):
            save_dir = f'/mnt/disks/disk/cifar10_test_layer_output{self.n_epoch}/{layer_idx}'
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            with open(f'{save_dir}/{self.batch_idx}.pt', 'wb') as f:
                torch.save(o, f)
        self.batch_idx += 1
    
    def on_epoch_end(self, epoch, **kwargs):
        self.n_epoch = epoch+1
