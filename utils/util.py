import time
import yaml
import torch
import os
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v
            
            
def readYamlConfig(configFileName):
    with open(configFileName) as f:
        data=yaml.safe_load(f)
        return data
    
from sklearn.metrics import roc_auc_score

def computeAUROC(scores,gt_list,obj,name="base"):
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print(obj + " image"+str(name)+" ROCAUC: %.3f" % (img_roc_auc))
    return img_roc_auc,img_scores  


def loadWeights(model,model_dir,alias):
    try:
        checkpoint = torch.load(os.path.join(model_dir, alias))
    except:
        raise Exception("Check saved model path.")
    model.load_state_dict(checkpoint["model"])
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
    return model


from datasets.mvtec import MVTecDataset
from datasets.visa import VisaDatasetAnomaly,VisaDatasetNormal
from datasets.mvtec3d import MVTec3dDataset
from datasets.eyecandies import EyecandiesDataset


def load_dataset(trainer):
    kwargs = ({"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {})
    
    if (trainer.dataset == "mvtec"):
        
        train_dataset = MVTecDataset(trainer.data_path,class_name=trainer.obj,
            is_train=True,resize=trainer.img_resize,cropsize=trainer.img_cropsize)
        
        img_nums = len(train_dataset)
        valid_num = int(img_nums * trainer.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_num, valid_num]
        )
        
        
        test_dataset = MVTecDataset(trainer.data_path,class_name=trainer.obj,
            is_train=False,resize=trainer.img_resize,cropsize=trainer.img_cropsize)
        
        
    if (trainer.dataset == "visa"):
        train_dataset = VisaDatasetNormal(trainer.data_path,class_name=trainer.obj,
            resize=trainer.img_resize,cropsize=trainer.img_cropsize)
        
        test_num=100
        train_val_num = len(train_dataset)-test_num
        
        valid_num = int(train_val_num * trainer.validation_ratio)
        train_num = train_val_num - valid_num
        train_data, val_data,test_dataNormal = torch.utils.data.random_split(
            train_dataset, [train_num, valid_num,test_num]
        )
        
        test_dataAnomaly = VisaDatasetAnomaly(trainer.data_path,class_name=trainer.obj,
            resize=trainer.img_resize,cropsize=trainer.img_cropsize)
        test_dataset = torch.utils.data.ConcatDataset([test_dataNormal,test_dataAnomaly])
        
    if (trainer.dataset == "mvtec3d"):
        
        train_dataset = MVTec3dDataset(trainer.data_path,class_name=trainer.obj,
            is_train=True,resize=trainer.img_resize,cropsize=trainer.img_cropsize)
        
        img_nums = len(train_dataset)
        valid_num = int(img_nums * trainer.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_num, valid_num]
        )
        
        test_dataset = MVTec3dDataset(trainer.data_path,class_name=trainer.obj,
            is_train=False,resize=trainer.img_resize,cropsize=trainer.img_cropsize)
    
    
    if (trainer.dataset == "eyecandies"):
        
        train_data = EyecandiesDataset(trainer.data_path,class_name=trainer.obj,
            split='train',image_index=0,resize=trainer.img_resize,cropsize=trainer.img_cropsize)
        
        val_data = EyecandiesDataset(trainer.data_path,class_name=trainer.obj,
            split='val',image_index=0,resize=trainer.img_resize,cropsize=trainer.img_cropsize)

        test_dataset = EyecandiesDataset(trainer.data_path,class_name=trainer.obj,
            split='test_public',image_index=0,resize=trainer.img_resize,cropsize=trainer.img_cropsize)
        
        
    trainer.train_loader = torch.utils.data.DataLoader(train_data, batch_size=trainer.batch_size, shuffle=True, **kwargs)
    trainer.val_loader = torch.utils.data.DataLoader(val_data, batch_size=trainer.batch_size, shuffle=True, **kwargs)


    trainer.train_examplar_loader = torch.utils.data.DataLoader(train_data, batch_size=trainer.n_embed, shuffle=True, **kwargs)
    trainer.val_examplar_loader = torch.utils.data.DataLoader(val_data, batch_size=trainer.n_embed, shuffle=True, **kwargs)
    trainer.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)