import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import InstanceNorm2d
# from models.basic_modules import *
# import models.My_TF as My_TF  
from basic_modules import *
import My_TF as My_TF  

class Rangebin_Attention(nn.Module):
    def __init__(self, input_dim, num_bins=4):
        super().__init__()
        self.coeff = input_dim ** -0.5
        self.q = Conv2dINReLU(input_dim, input_dim, (3, 11))
        self.k = Conv2dINReLU(input_dim, input_dim, (3, 11))
        self.v = Conv2dINReLU(input_dim, input_dim, (1, 1))
        self.proj = nn.Sequential(
            nn.Linear(num_bins, 4*num_bins),
            nn.LeakyReLU(0.3),
            nn.Linear(4*num_bins, 1)
        )
    def forward(self, x):
        # x: [B, 16, 4, 1024]
        q, k, v = self.q(x), self.k(x), self.v(x)   # [B, 16, 4, 1024]
        scores = q @ k.transpose(-2, -1)            # [B, 16, 4, 4]
        scores = F.softmax(scores * self.coeff, dim=-2)
        h = scores @ v                              # [B, 16, 4, 1024]
        h = self.proj(h.transpose(-2, -1)).transpose(-2, -1)
        return h    
    
class UNet_Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, F_norm=nn.BatchNorm2d, dropout=0.0, last_conv=False):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        if last_conv == False:
            self.layer = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
                F_norm(C_out),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.3, inplace=True),

                nn.Conv2d(C_out, C_out, kernel_size,
                          stride=1, padding=padding, bias=False),
                F_norm(C_out),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.3, inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
                F_norm(C_out),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(C_out, C_out, kernel_size,
                          stride=1, padding=padding, bias=False),
            )

    def forward(self, x):
        return self.layer(x)


class Down_Conv(nn.Module):
    def __init__(self, C, kernel_size, stride, F_norm=nn.BatchNorm2d):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, kernel_size, stride=stride, padding=padding, bias=False),
            F_norm(C),
            nn.LeakyReLU(0.3, inplace=True)
        )

    def forward(self, x):
        return self.Down(x)


class Up_Conv(nn.Module):
    def __init__(self, C, kernel_size, stride, output_padding, F_norm=nn.BatchNorm2d):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(kernel_size[i] - stride[i] + output_padding[i]
                        ) // 2 for i in range(len(kernel_size))]
        else:
            padding = (kernel_size - stride + output_padding) // 2
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(C, C//2, kernel_size, stride,
                               padding, output_padding, bias=False),
            F_norm(C//2),
            nn.LeakyReLU(0.3, inplace=True)
        )

    def forward(self, x, r):
        return torch.cat((self.Up(x), r), 1)
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, mlp_ratio=4., nhead=16, num_layers=12):
        super().__init__()
        dim_feedforward = int(d_model * mlp_ratio)
        encoder_layer = My_TF.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0, activation='gelu',
                                                      batch_first=True, norm_first=True)
        self.tf_encoder = My_TF.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, -2, -1)  # [B, seq, #features]
        x = self.tf_encoder(x)
        x = torch.transpose(x, -2, -1)  # [B, #features, seq]
        x = torch.unsqueeze(x, dim=2)
        return x
    
class UNet_Transformer_InsNorm_Attn_Cut_KD_MyTF_Multi_Expert(nn.Module):
    ### WaveBP Type I
    def __init__(self, gt_norm=False, num_bins=4):
        super().__init__()
        self.gt_norm = gt_norm
        self.iq_conv = nn.Sequential(
            Conv(2, 16, 1, 1),
            InstanceNorm2d(16),
            nn.LeakyReLU(0.3)
        )  # [B, 16, 4, 1024]
        self.rangebin_weight0 = Rangebin_Attention(16, num_bins)  # [B, 32, 1, 1024]
        self.C1 = UNet_Conv(16, 32, (1, 11), F_norm=InstanceNorm2d)  # [B, 32, 4, 1024]
        self.rangebin_weight1 = Rangebin_Attention(32, num_bins)  # [B, 32, 1, 1024]
        self.D1 = Down_Conv(32, (1, 11), (1, 2), F_norm=InstanceNorm2d)  # [B, 32, 4, 512]
        self.C2 = UNet_Conv(32, 64, (1, 11), F_norm=InstanceNorm2d)  # [B, 64, 1, 512]
        self.rangebin_weight2 = Rangebin_Attention(64, num_bins)  # [B, 64, 1, 512]
        self.D2 = Down_Conv(64, (1, 11), (1, 2), F_norm=InstanceNorm2d)  # [B, 64, 1, 256]
        self.C3 = UNet_Conv(64, 128, (1, 11), F_norm=InstanceNorm2d)
        self.rangebin_weight3 = Rangebin_Attention(128, num_bins)  # [B, 32, 1, 256]
        self.D3 = Down_Conv(128, (1, 11), (1, 2), F_norm=InstanceNorm2d)  # [B, 128, 1, 128]
        self.C4 = UNet_Conv(128, 256, (1, 11), F_norm=InstanceNorm2d)
        self.D4 = Down_Conv(256, (1, 11), (1, 2), F_norm=InstanceNorm2d)  # [B, 256, 1, 64]
        self.rangebin_weight4 = Rangebin_Attention(256, num_bins)
        self.transformer = TransformerEncoder(
            d_model=256, mlp_ratio=4., nhead=8, num_layers=12)  # [B, 256, 1, 64]

        self.U1 = Up_Conv(256, (1, 11), (1, 2), (0, 1), F_norm=InstanceNorm2d)  # [B, 256, 1, 128]
        self.C5 = UNet_Conv(256, 128, (1, 11), F_norm=InstanceNorm2d)  # [B, 128, 1, 128]
        self.U2 = Up_Conv(128, (1, 11), (1, 2), (0, 1), F_norm=InstanceNorm2d)  # [B, 128, 1, 256]
        self.C6 = UNet_Conv(128, 64, (1, 11), F_norm=InstanceNorm2d)  # [B, 64, 1, 256]
        self.U3 = Up_Conv(64, (1, 11), (1, 2), (0, 1), F_norm=InstanceNorm2d)  # [B, 64, 1, 512]
        self.C7 = UNet_Conv(64, 32, (1, 11), F_norm=InstanceNorm2d)  # [B, 32, 1, 512]
        self.U4 = Up_Conv(32, (1, 11), (1, 2), (0, 1), F_norm=InstanceNorm2d)  # [B, 32, 1, 1024]
        self.C8 = UNet_Conv(32, 16, (1, 11), F_norm=InstanceNorm2d)  # [B, 16, 1, 1024]

        self.expert1 = nn.Sequential(
            nn.Conv1d(16, 64, 11, 1, 5, bias=False),   # [B, 1, 1024],
            nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
        self.expert2 = nn.Sequential(
            nn.Conv1d(16, 64, 11, 1, 5, bias=False),   # [B, 1, 1024],
            nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
        self.expert3 = nn.Sequential(
            nn.Conv1d(16, 64, 11, 1, 5, bias=False),   # [B, 1, 1024],
            nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
        self.expert4 = nn.Sequential(
            nn.Conv1d(16, 64, 11, 1, 5, bias=False),   # [B, 1, 1024],
            nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )

        self.expert5 = nn.Sequential(
            nn.Conv1d(16, 64, 11, 1, 5, bias=False),   # [B, 1, 1024],
            nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),   # [B, 1, 1024]
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
        self.experts = [self.expert1, self.expert2, self.expert3, self.expert4, self.expert5]
        if gt_norm:
            self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, bp_cate, return_all_expert_results=False):
        # pi: sex / height / age / weight, [B, 4]
        if isinstance(inputs, list):
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs
        x = self.iq_conv(x)  # [B, 16, 4, 1024]
        attn_cut0 = self.rangebin_weight0(x)  # [B, 16, 1 ,1024]
        DS1 = self.D1(self.C1(x))  # [B, 32, 1, 512]
        attn_cut1 = self.rangebin_weight1(DS1)
        DS2 = self.D2(self.C2(DS1))  # [B, 64, 1, 1024]
        attn_cut2 = self.rangebin_weight2(DS2)
        DS3 = self.D3(self.C3(DS2))  # [B, 128, 1, 512]
        attn_cut3 = self.rangebin_weight3(DS3)
        DS4 = self.D4(self.C4(DS3))  # [B, 256, 1, 256]
        DS4 = self.rangebin_weight4(DS4)
        DS4 = self.transformer(DS4)  # [B, 256, 1, 256]
        UP1 = self.C5(self.U1(DS4, attn_cut3))  # [B, 128, 1, 512]
        UP2 = self.C6(self.U2(UP1, attn_cut2))  # [B, 64, 1, 1024]
        UP3 = self.C7(self.U3(UP2, attn_cut1))  # [B, 32, 1, 512]
        UP4 = self.C8(self.U4(UP3, attn_cut0))  # [B, 16, 1, 1024]
        UP4 = torch.squeeze(UP4, dim=2)
        if return_all_expert_results == False:
            out = torch.zeros((x.shape[0], 1, x.shape[-1])).to(x.device)
            for category in range(1, 6):
                cate_indices = torch.where(bp_cate == category)[0]
                if len(cate_indices) == 0:
                    continue
                if len(cate_indices) == 1:
                    out[cate_indices:cate_indices+1, ...] = self.experts[category - 1](UP4[cate_indices:cate_indices+1, ...])
                else:
                    out[cate_indices] = self.experts[category - 1](UP4[cate_indices, ...])
        else:
            out = torch.zeros((x.shape[0], 5, x.shape[-1])).to(x.device)
            for category in range(0, 5):
                out[:, category, :] = self.experts[category](UP4).squeeze()
        if self.gt_norm:
            return self.sigmoid(out), torch.squeeze(UP4, dim=2)
        else:
            return out, torch.squeeze(UP4, dim=2)
        

        

class UNet_Transformer_InsNorm_Attn_Cut_KD_MyTF_Multi_Expert_Extractor_For_Finetune(nn.Module):
    ### WaveBP Type II
    def __init__(self, gt_norm=False, num_bins=4):
        super().__init__()
        self.gt_norm = gt_norm
        self.iq_conv = nn.Sequential(
            Conv(2, 16, 1, 1),
            InstanceNorm2d(16),
            nn.LeakyReLU(0.3)
        )  
        self.rangebin_weight0 = Rangebin_Attention(16, num_bins)  
        self.C1 = UNet_Conv(16, 32, (1, 11), F_norm=InstanceNorm2d)  
        self.rangebin_weight1 = Rangebin_Attention(32, num_bins)  
        self.D1 = Down_Conv(32, (1, 11), (1, 2), F_norm=InstanceNorm2d)  
        self.C2 = UNet_Conv(32, 64, (1, 11), F_norm=InstanceNorm2d)  
        self.rangebin_weight2 = Rangebin_Attention(64, num_bins)  
        self.D2 = Down_Conv(64, (1, 11), (1, 2), F_norm=InstanceNorm2d)  
        self.C3 = UNet_Conv(64, 128, (1, 11), F_norm=InstanceNorm2d)
        self.rangebin_weight3 = Rangebin_Attention(128, num_bins) 
        self.D3 = Down_Conv(128, (1, 11), (1, 2), F_norm=InstanceNorm2d)  
        self.C4 = UNet_Conv(128, 256, (1, 11), F_norm=InstanceNorm2d)
        self.D4 = Down_Conv(256, (1, 11), (1, 2), F_norm=InstanceNorm2d)  
        self.rangebin_weight4 = Rangebin_Attention(256, num_bins)
        self.transformer = TransformerEncoder(d_model=256, mlp_ratio=4., nhead=8, num_layers=12)  
        self.U1 = Up_Conv(256, (1, 11), (1, 2), (0, 1), F_norm=InstanceNorm2d)  
        self.C5 = UNet_Conv(256, 128, (1, 11), F_norm=InstanceNorm2d) 
        self.U2 = Up_Conv(128, (1, 11), (1, 2), (0, 1), F_norm=InstanceNorm2d)
        self.C6 = UNet_Conv(128, 64, (1, 11), F_norm=InstanceNorm2d)
        self.U3 = Up_Conv(64, (1, 11), (1, 2), (0, 1), F_norm=InstanceNorm2d) 
        self.C7 = UNet_Conv(64, 32, (1, 11), F_norm=InstanceNorm2d) 
        self.U4 = Up_Conv(32, (1, 11), (1, 2), (0, 1), F_norm=InstanceNorm2d)  
        self.C8 = UNet_Conv(32, 16, (1, 11), F_norm=InstanceNorm2d)  
        
        self.expert1 = nn.Sequential(
            nn.Conv1d(16, 64, 11, 1, 5, bias=False),  
            nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, 1, 5, bias=False),   
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),   
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
    def forward(self, x):
        x = self.iq_conv(x)  
        attn_cut0 = self.rangebin_weight0(x)  
        DS1 = self.D1(self.C1(x))  
        attn_cut1 = self.rangebin_weight1(DS1)
        DS2 = self.D2(self.C2(DS1))  
        attn_cut2 = self.rangebin_weight2(DS2)
        DS3 = self.D3(self.C3(DS2))  
        attn_cut3 = self.rangebin_weight3(DS3)
        DS4 = self.D4(self.C4(DS3))  
        DS4 = self.rangebin_weight4(DS4)
        DS4 = self.transformer(DS4) 
        UP1 = self.C5(self.U1(DS4, attn_cut3))  
        UP2 = self.C6(self.U2(UP1, attn_cut2))  
        UP3 = self.C7(self.U3(UP2, attn_cut1))  
        UP4 = self.C8(self.U4(UP3, attn_cut0))  
        UP4 = torch.squeeze(UP4, dim=2)
        out = self.expert1(UP4)
        if self.gt_norm:
            return self.sigmoid(out), torch.squeeze(UP4, dim=2)
        else:
            return out, torch.squeeze(UP4, dim=2)
        

if __name__ == '__main__':
    import time
    import numpy as np
    x = torch.rand((1, 2, 4, 1024))
    model = UNet_Transformer_InsNorm_Attn_Cut_KD_MyTF_Multi_Expert().cuda()
    m = []
    t = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(1000):
        starter.record()
        y = model(x.cuda(), torch.Tensor(0))
        ender.record()
        torch.cuda.synchronize()
        t.append(starter.elapsed_time(ender) / 1000)
        allocated_memory = torch.cuda.max_memory_allocated()
        
        m.append(allocated_memory/1024**2)
    
    print(np.mean(m))
    print(np.mean(t))
    