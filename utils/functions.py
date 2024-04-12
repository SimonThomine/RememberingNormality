import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

def cal_loss(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
 
        f_loss = 0.5 * (ft_norm - fs_norm) ** 2
        t_loss += f_loss.sum() / (h * w)

    return t_loss / N

def cal_loss_cosine(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        f_loss = 1 - F.cosine_similarity(fs_norm, ft_norm)
        t_loss += f_loss.sum()/ (h * w)
    return t_loss / N


def cal_loss_orth(student):
    keys=[student.memory0.keys,student.memory1.keys,student.memory2.keys]
    values=[student.memory0.values,student.memory1.values,student.memory2.values]
    t_loss=0
    for key,value in zip(keys,values):
        key_norm = torch.nn.functional.normalize(key, dim=1)
        value_norm = torch.nn.functional.normalize(value, dim=1)
        cos_sim = torch.mm(key_norm, value_norm.T) 
        t_loss += cos_sim.sum()/key.size(0)**2
    return t_loss


@torch.no_grad()
def cal_anomaly_maps(fs_list, ft_list, out_size):
    anomaly_map = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)

        _, _, h, w = fs.shape

        a_map = (0.5 * (ft_norm - fs_norm) ** 2) / (h * w)

        a_map = a_map.sum(1, keepdim=True)

        a_map = F.interpolate(
            a_map, size=out_size, mode="bilinear", align_corners=False
        )
        anomaly_map += a_map
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

    return anomaly_map

