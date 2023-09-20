""" 
EuroSAT dataset loader
""" 
import torch
from torchvision.datasets import ImageFolder

def random_split(dataset, ratio=0.9, random_state=None):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split


class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None, download = False):
        if download : 
            self.download(root)
        
        super().__init__(root, transform=transform, target_transform=target_transform)
