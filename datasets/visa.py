import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T



class VisaDatasetNormal(Dataset):
    def __init__(self, dataset_path='../datasets', class_name='candle',resize=256, cropsize=256):
        self.dataset_path = dataset_path
        self.class_name = class_name

        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y, self.mask = self.load_dataset_folder()
        
        self.transform_x = T.Compose([T.Resize(resize),                   
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])
        
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)
    
    def load_dataset_folder(self):
        x, y, mask = [], [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, "Data/Images/Normal")
        
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if (f.endswith('.JPG')or f.endswith('.png'))])
        
        x.extend(img_fpath_list)
        y.extend([0] * len(img_fpath_list))
        mask.extend([None] * len(img_fpath_list))
                
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
     
     
     
# TODO visa Dataset anomalous
class VisaDatasetAnomaly(Dataset):
    def __init__(self, dataset_path='../datasets', class_name='candle',resize=256, cropsize=256):
        self.dataset_path = dataset_path
        self.class_name = class_name

        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y, self.mask = self.load_dataset_folder()
        
        self.transform_x = T.Compose([T.Resize(resize),                   
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])
        
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)
    
    def load_dataset_folder(self):
        x, y, mask = [], [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, "Data/Images/Anomaly")
        mask_dir = os.path.join(self.dataset_path, self.class_name, "Data/Masks/Anomaly")
        
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if (f.endswith('.JPG')or f.endswith('.png'))])
        mask_fpath_list = sorted([os.path.join(mask_dir, f)
                                     for f in os.listdir(mask_dir)
                                     if (f.endswith('.JPG')or f.endswith('.png'))]) 
        
        x.extend(img_fpath_list)
        y.extend([1] * len(img_fpath_list))
        mask.extend(mask_fpath_list)
                
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
    

