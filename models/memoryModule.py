import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F 

class memoryModule(nn.Module):
    def __init__(self,L=50,channel=128, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.keys=torch.randn(L,channel).to('cuda')
        self.values=torch.randn(L,channel).to('cuda')
        
        self.softmax = nn.Softmax(dim=1)
        self.channel=channel
    
    # Input of size (b,C,H,W) 
    def forward(self, x: Tensor,normality=False) -> Tensor:
        b,c,h,w=x.size()
        x=x.view(x.size(0),x.size(1),-1).permute(0,2,1)
        
        if (normality==False):
            # The softmax is computed with the keys
            keysAct = self.keys.unsqueeze(0).expand(x.size(0), -1, -1)
            norm_keys = F.normalize(keysAct, p=2, dim=2)
            norm_x = F.normalize(x, p=2, dim=2)
            cos_sim = torch.matmul(norm_keys.unsqueeze(2), norm_x.transpose(1, 2).unsqueeze(1)).squeeze(2)
            sim_vec=self.softmax(cos_sim)
            valuesAct = self.values.unsqueeze(0).expand(x.size(0), -1, -1)
        else : 
            # The softmax is computed with the values
            valuesAct = self.values.unsqueeze(0).expand(x.size(0), -1, -1)
            norm_values = F.normalize(valuesAct, p=2, dim=2)
            norm_x = F.normalize(x, p=2, dim=2)
            cos_sim = torch.matmul(norm_values.unsqueeze(2), norm_x.transpose(1, 2).unsqueeze(1)).squeeze(2)
            sim_vec=self.softmax(cos_sim)
        
        Fnorm=torch.matmul(sim_vec.permute(0,2,1), valuesAct)
        Fr=Fnorm.permute(0,2,1).view(b,self.channel,h,w)    
            
        return Fr
    





#! Memory module test
# L=50 # was 50
# softmax = nn.Softmax(dim=1)
# x = torch.randn(16,128,32,32)
# b,c,h,w=x.size()
# keys = torch.randn(L, 128) 
# values=torch.randn(L,128)
# x=x.view(x.size(0),x.size(1),-1).permute(0,2,1)

# keys = keys.unsqueeze(0).expand(x.size(0), -1, -1)


# norm_keys = F.normalize(keys, p=2, dim=2)
# norm_x = F.normalize(x, p=2, dim=2)
# cos_sim = torch.matmul(norm_keys.unsqueeze(2), norm_x.transpose(1, 2).unsqueeze(1)).squeeze()

# sim_vec = softmax(cos_sim)


# values = values.unsqueeze(0).expand(x.size(0), -1, -1)
# Fnorm=torch.matmul(sim_vec.permute(0,2,1), values)
# Fr=Fnorm.permute(0,2,1).view(16,128,h,w)
# print(Fr.shape)


