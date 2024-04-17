import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import yaml

class EyecandiesDataset(Dataset):
    def __init__(self, dataset_path='../datasets', class_name='bottle', split='train',image_index=0,
                 resize=256, cropsize=256): 
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.split=split
        self.image_index=image_index    
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

        img_dir = os.path.join(self.dataset_path, self.class_name, self.split,'data')
        


        img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if (f.endswith(str(self.image_index)+'.png'))])
        msk_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if (f.endswith('mask.png') and 'bumps'not in f and 'colors' not in f 
                                        and 'normal' not in f and 'dents' not in f)])
        
        x.extend(img_fpath_list)
        mask.extend(msk_fpath_list)

        for f in sorted(os.listdir(img_dir)) :
            if (f.endswith('metadata.yaml')):
                with open(os.path.join(img_dir, f), "r") as yaml_file:
                    data = yaml.safe_load(yaml_file)
                    if (any(value == 1 for value in data.values()))==True:
                        y.extend([1])
                    else:
                        y.extend([0])
                        
        assert len(x) == len(y), 'number of x and y should be same'
        assert len(x) == len(mask), 'number of x and mask should be same'
        return list(x), list(y), list(mask)