import numpy as np
from PIL import Image

from torch.utils import data
import torch
from torchvision.transforms import ToTensor

class VisDataset(data.Dataset):

    def __init__(self, filename_tuples):
        self.filenames = filename_tuples

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_ref = Image.open(filename[0]).convert('RGB') #1-channel to 3-channel
        img_ref = np.asarray(img_ref).copy()
        img_def = Image.open(filename[1]).convert('RGB')
        img_def = np.asarray(img_def).copy()

        img_ref = ToTensor()(img_ref) #converts [0,255] to [0.0,1.0]
        img_def = ToTensor()(img_def)
        img = torch.stack((img_ref, img_def), dim=1) #stack to create a 5d-tensor. ref[:,:,0,:,:]. def[:,:,0,:,:]
        
        return img.contiguous(), filename[1]