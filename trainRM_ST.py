import os
import time
import numpy as np
import torch
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from utils.util import  AverageMeter,readYamlConfig,computeAUROC,loadWeights
from utils.functions import (
    cal_loss,
    cal_loss_cosine,
    cal_loss_orth,
    cal_anomaly_maps
)
from models.ST.teacherST import wide_resnet50_2,resnet18
from models.ST.resnetRM import resnet18Memory,wide_resnet50_2Memory


class NetTrainer:          
    def __init__(self, data,device):  
        self.device = device
        self.validation_ratio = 0.2
        self.data_path = data['data_path']
        self.obj = data['obj']
        self.img_resize = data['TrainingData']['img_size']
        self.img_cropsize = data['TrainingData']['crop_size']
        self.num_epochs = data['TrainingData']['epochs']
        self.lr = data['TrainingData']['lr']
        self.batch_size = data['TrainingData']['batch_size'] 
        self.embedDim = data['TrainingData']['embedDim']
        self.n_embed = data['TrainingData']['n_embed']
        self.lambda1 = data['TrainingData']['lambda1']
        self.lambda2 = data['TrainingData']['lambda1']
        self.model_dir = data['save_path'] + "/models" + "/" + self.obj
        os.makedirs(self.model_dir, exist_ok=True)
        self.modelName = data['backbone']
                        
        self.load_model()
        self.load_dataset()
        
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999)) 
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.lr*10,epochs=self.num_epochs,steps_per_epoch=len(self.train_loader))
        

    def load_dataset(self):
        kwargs = (
            {"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        train_dataset = MVTecDataset(
            self.data_path,
            class_name=self.obj,
            is_train=True,
            resize=self.img_resize,
            cropsize=self.img_cropsize,
        )
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_num, valid_num]
        )
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True, **kwargs)

    
        self.train_examplar_loader = torch.utils.data.DataLoader(train_data, batch_size=self.n_embed, shuffle=True, **kwargs)
        self.val_examplar_loader = torch.utils.data.DataLoader(val_data, batch_size=self.n_embed, shuffle=True, **kwargs)


    def load_model(self):
        print("loading and training SingleNet")

        if self.modelName == "resnet18":
            self.teacher=resnet18(pretrained=True)
            self.student=resnet18Memory(embedDim=self.embedDim).to(self.device)
        elif self.modelName == "wide_resnet50_2":
            self.teacher=wide_resnet50_2(pretrained=True).to(self.device)
            self.student=wide_resnet50_2Memory(embedDim=self.embedDim).to(self.device)
        else : 
            print("Invalid/unconfigured model name")
            exit()
        
        self.teacher=self.teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self):
        print("training " + self.obj)
        self.student.train()
                
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs,desc="Training",unit="batch")
        
        for _ in range(1, self.num_epochs + 1):
            losses = AverageMeter()
            for (image,_, _),(imageExamplar,_,_) in zip(self.train_loader, self.train_examplar_loader):
                image= image.to(self.device)
                imageExamplar= imageExamplar.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):

                    features_s,features_t,features_t_examplar,features_t_examplar_norm  = self.infer(image,imageExamplar) 
                    
                    loss_KD=cal_loss_cosine(features_s, features_t)
                    loss_NM=cal_loss(features_t_examplar, features_t_examplar_norm)
                    loss_ORTH=cal_loss_orth(self.student)
                    
                    loss=loss_KD+self.lambda1*loss_NM +self.lambda2*loss_ORTH 
                    
                    losses.update(loss.sum().item(), image.size(0))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()
            
            val_loss = self.val(epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()

            epoch_time.update(time.time() - start_time)
            start_time = time.time()
        epoch_bar.close()
        
        print("Training end.")

    def val(self, epoch_bar):
        self.student.eval()
        losses = AverageMeter()
        for (image,_, _),(imageExamplar,_,_) in zip(self.val_loader,self.val_examplar_loader):
            image= image.to(self.device)
            imageExamplar= imageExamplar.to(self.device)
            with torch.set_grad_enabled(False):
                
                features_s,features_t,features_t_examplar,features_t_examplar_norm  = self.infer(image,imageExamplar)  
                
                
                loss_KD=cal_loss_cosine(features_s, features_t)
                loss_NM=cal_loss(features_t_examplar, features_t_examplar_norm)
                loss_ORTH=cal_loss_orth(self.student)
                    
                loss=loss_KD+self.lambda1*loss_NM +self.lambda2*loss_ORTH 
                
                
                losses.update(loss.item(), image.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))


    @torch.no_grad()
    def test(self):

        self.student=loadWeights(self.student,self.model_dir,"student.pth")
        
        kwargs = (
            {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        test_dataset = MVTecDataset(
            self.data_path,
            class_name=self.obj,
            is_train=False,
            resize=self.img_resize,
            cropsize=self.img_cropsize,
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        scores = []
        test_imgs = []
        gt_list = []
        progressBar = tqdm(test_loader)
        for image, label, _ in test_loader:
            test_imgs.extend(image.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            
            image = image.to(self.device)
            with torch.set_grad_enabled(False):
                features_t = self.teacher(image)
                features_s=self.student(image)
                
                score =cal_anomaly_maps(features_s,features_t,self.img_cropsize) 
                
                progressBar.update()
                
            scores.append(score)

        progressBar.close()
        scores = np.asarray(scores)
        gt_list = np.asarray(gt_list)
        img_roc_auc,_=computeAUROC(scores,gt_list,self.obj,"singleNet")

        return img_roc_auc
    
    def infer(self, img,imgExamplar):
        
        features_t_examplar = self.teacher.forward_normality_embedding(imgExamplar)
        features_t_examplar_norm=[self.student.memory0(features_t_examplar[0]),
                                  self.student.memory1(features_t_examplar[1]),
                                  self.student.memory2(features_t_examplar[2])]

        features_t = self.teacher(img)
        features_s=self.student(img)

        return features_s,features_t ,features_t_examplar,features_t_examplar_norm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=readYamlConfig("config.yaml")
    distill = NetTrainer(data,device)
     
    if data['phase'] == "train":
        distill.train()
        distill.test()
    elif data['phase'] == "test":
        distill.test()
    else:
        print("Phase argument must be train or test.")

