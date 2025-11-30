import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from models.basic_modules import *
import models.My_TF as My_TF 
class UNet_Conv1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, F_norm=nn.BatchNorm1d, last_conv=False, dropout_ratio=0.0):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        if last_conv == False:
            self.layer = nn.Sequential(
                nn.Conv1d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(C_out),
                # SwitchNorm2d(C_out),
                F_norm(C_out),
                # F_norm,
                nn.Dropout(dropout_ratio),
                nn.LeakyReLU(0.3, inplace=True),

                nn.Conv1d(C_out, C_out, kernel_size,
                          stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(C_out),
                # SwitchNorm2d(C_out),
                F_norm(C_out),
                # F_norm,
                nn.Dropout(dropout_ratio),
                nn.LeakyReLU(0.3, inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv1d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(C_out),
                # SwitchNorm2d(C_out),
                F_norm(C_out),
                # F_norm,
                nn.Dropout(dropout_ratio),
                nn.LeakyReLU(0.3, inplace=True),

                nn.Conv1d(C_out, C_out, kernel_size,
                          stride=1, padding=padding, bias=False),

            )

    def forward(self, x):
        return self.layer(x)

class Down_Conv1D(nn.Module):
    def __init__(self, C, kernel_size, stride, F_norm=nn.BatchNorm1d):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        self.Down = nn.Sequential(
            nn.Conv1d(C, C, kernel_size, stride=stride, padding=padding, bias=False),
            # SwitchNorm2d(C),
            F_norm(C),
            # F_norm,
            nn.LeakyReLU(0.3, inplace=True)
        )

    def forward(self, x):
        return self.Down(x)


class Up_Conv1D(nn.Module):
    def __init__(self, C, kernel_size, stride, output_padding, F_norm=nn.BatchNorm1d):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(kernel_size[i] - stride[i] + output_padding[i]
                        ) // 2 for i in range(len(kernel_size))]
        else:
            padding = (kernel_size - stride + output_padding) // 2
        self.Up = nn.Sequential(
            nn.ConvTranspose1d(C, C//2, kernel_size, stride,
                               padding, output_padding, bias=False),
            # SwitchNorm2d(C//2),
            F_norm(C//2),
            # F_norm,
            nn.LeakyReLU(0.3, inplace=True)
        )

    def forward(self, x, r):
        return torch.cat((self.Up(x), r), 1)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, mlp_ratio=4., nhead=16, num_layers=12):
        super().__init__()
        dim_feedforward = int(d_model * mlp_ratio)
        encoder_layer = My_TF.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.0, activation='gelu',
                                                      batch_first=True, norm_first=True)
        self.tf_encoder = My_TF.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, #feats, seq]
        x = torch.transpose(x, -2, -1)  # [B, seq, #features]
        x = self.tf_encoder(x)
        x = torch.transpose(x, -2, -1)  # [B, #features, seq]
        return x
    
class UNet_Transformer_InsNorm_1D_Multi_Expert(nn.Module):
    def __init__(self, gt_norm=False):
        super().__init__()
        self.gt_norm = gt_norm
        # input: [B, 2, 1024]
        self.ECG_PPG_conv = nn.Sequential(
            nn.Conv1d(2, 16, 1, 1),
            # nn.BatchNorm1d(16)
            nn.InstanceNorm1d(16),
            nn.LeakyReLU(0.3),
        )  # [B, 16, 1024]
        self.C1 = UNet_Conv1D(16, 32, 11, F_norm=nn.InstanceNorm1d)  # [B, 32, 1024]
        self.D1 = Down_Conv1D(32, 11, 2, F_norm=nn.InstanceNorm1d)  # [B, 32, 512]
        self.C2 = UNet_Conv1D(32, 64, 11, F_norm=nn.InstanceNorm1d)  # [B, 64, 512]
        self.D2 = Down_Conv1D(64, 11, 2, F_norm=nn.InstanceNorm1d)  # [B, 64, 1024]
        self.C3 = UNet_Conv1D(64, 128, 11, F_norm=nn.InstanceNorm1d)
        self.D3 = Down_Conv1D(128, 11, 2, F_norm=nn.InstanceNorm1d)  # [B, 128, 512]
        self.C4 = UNet_Conv1D(128, 256, 11, F_norm=nn.InstanceNorm1d)
        self.D4 = Down_Conv1D(256, 11, 2, F_norm=nn.InstanceNorm1d)  # [B, 256, 256]
        self.transformer = TransformerEncoder(
            d_model=256, mlp_ratio=4., nhead=8, num_layers=12)  # [B, 256, 256]

        self.U1 = Up_Conv1D(256, 11, 2, 1, F_norm=nn.InstanceNorm1d)  # [B, 256, 512]
        self.C5 = UNet_Conv1D(256, 128, 11, F_norm=nn.InstanceNorm1d)  # [B, 128, 512]
        self.U2 = Up_Conv1D(128, 11, 2, 1, F_norm=nn.InstanceNorm1d)  # [B, 128, 1024]
        self.C6 = UNet_Conv1D(128, 64, 11, F_norm=nn.InstanceNorm1d)  # [B, 64, 1024]
        self.U3 = Up_Conv1D(64, 11, 2, 1, F_norm=nn.InstanceNorm1d)  # [B, 64, 512]
        self.C7 = UNet_Conv1D(64, 32, 11, F_norm=nn.InstanceNorm1d)  # [B, 32, 512]
        self.U4 = Up_Conv1D(32, 11, 2, 1, F_norm=nn.InstanceNorm1d)  # [B, 32, 1024]
        self.C8 = UNet_Conv1D(32, 16, 11, F_norm=nn.InstanceNorm1d)  # [B, 16, 1024]
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
        # self.experts = [self.expert1, self.expert2, self.expert3, self.expert4]
        self.experts = [self.expert1, self.expert2, self.expert3, self.expert4, self.expert5]
        if gt_norm:
            self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, bp_cate, return_all_expert_results=False):
        if isinstance(inputs, list):
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs
        x = self.ECG_PPG_conv(x)  # [B, 16, 1024]
        DS1 = self.D1(self.C1(x))  # [B, 32, 512]
        DS2 = self.D2(self.C2(DS1))  # [B, 64, 256]
        DS3 = self.D3(self.C3(DS2))  # [B, 128, 128]
        DS4 = self.D4(self.C4(DS3))  # [B, 256, 64]
        DS4 = self.transformer(DS4)  # [B, 256, 64]

        UP1 = self.C5(self.U1(DS4, DS3))  # [B, 128, 1, 128]

        UP2 = self.C6(self.U2(UP1, DS2))  # [B, 64, 1, 256]
        UP3 = self.C7(self.U3(UP2, DS1))  # [B, 32, 1, 512]
        UP4 = self.C8(self.U4(UP3, x))  # [B, 16, 1, 1024]
        if return_all_expert_results == False:
            out = torch.zeros((x.shape[0], 1, x.shape[2])).to(x.device)
            if bp_cate is not None:
                for category in range(1, 6):
                    cate_indices = torch.where(bp_cate == category)[0]
                    if len(cate_indices) == 0:
                        continue
                    if len(cate_indices) == 1:
                        out[cate_indices:cate_indices+1, ...] = self.experts[category - 1](UP4[cate_indices:cate_indices+1, ...])
                    else:
                        out[cate_indices] = self.experts[category - 1](UP4[cate_indices, ...])
            else:
                for category in range(1, 6):
                    out += self.experts[category-1](UP4)
                out = out / 5        
        else:
            out = torch.zeros((x.shape[0], 5, x.shape[-1])).to(x.device)
            for category in range(0, 5):
                out[:, category, :] = self.experts[category](UP4).squeeze()
        if self.gt_norm:
            return self.sigmoid(out), torch.squeeze(UP4)
        else:
            return out, torch.squeeze(UP4)
