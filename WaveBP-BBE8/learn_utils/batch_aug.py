import torch
from torch.distributions.beta import Beta
import numpy as np
import math
from dataset.normalizer import normalizer
'''
augment the train data in the same batch(numpy class)
'''

def beamforming(x, theta=0, beam_norm=False):
    '''
    x: [Batch, 2, 4, 4, 1024] -> [Batch, I/Q, 4Rx, 4RangeBins, Time]
    y: [Batch, 1, 1024]
    theta: in rad
    '''
    c = 3 * 1e8
    fc = 79 * 1e9
    wave_len = c/fc
    rx_pos = np.array([0*wave_len, 0.5*wave_len, 1*wave_len, 1.5*wave_len])
    rx_pos = rx_pos / wave_len
    coeffs = np.exp(-1j*2*np.pi*rx_pos*np.sin(theta))
    coeffs = np.conj(coeffs)    
    x = np.transpose(x, axes=(0, 1, 3, 4, 2))   # [Batch, I/Q, 4RangeBins, Time, 4Tx]
    beam_x = np.matmul(x, coeffs)    # [Batch, I/Q, 4RangeBins, Time]
    return beam_x

def beamforming_torch(x, theta=0):
    '''
    x: Tensor [Batch, 2, 4, 4, 1024] -> [Batch, I/Q, 4Rx, 4RangeBins, Time]
    y: Tensor [Batch, 1, 1024]
    theta: in rad
    '''
    c = 3 * 1e8
    fc = 79 * 1e9
    wave_len = c/fc
    rx_pos = torch.Tensor([0*wave_len, 0.5*wave_len, 1*wave_len, 1.5*wave_len])
    rx_pos = rx_pos / wave_len
    coeffs = torch.exp(-1j*2*np.pi*rx_pos*np.sin(theta))
    coeffs = torch.conj(coeffs)
    coeffs = coeffs.to(x.device)
    x_complex = x[:, 0, ...] + 1j*x[:, 1, ...]  # [Batch, 4Rx, 4RangeBins, Time]
    x_complex = x_complex.permute(0, 2, 3, 1)   # [Batch, 4RangeBins, Time, 4Rx]
    beam_x = torch.matmul(x_complex, coeffs)    # [Batch, 4RangeBins, Time]
    y = torch.zeros(beam_x.shape[0], 2, beam_x.shape[1], beam_x.shape[2])
    y[:, 0, ...] = beam_x.real
    y[:, 1, ...] = beam_x.imag
    return y


def beamform_aug(x, start_angle=-30, end_angle=31, step=5, beam_norm=False):
    '''
    x: Tensor [Batch, 2, 4Rx, 4RBs, 1024]
    aug_x_list: [[Batch, 2, 4RBs, 1024]..., ]
    '''
    aug_x_list = []
    num_chs = x.shape[2]
    for i in range(num_chs):
        aug_x_list.append(x[:, :, i, :, :]) # only one antenna
    for angle in range(start_angle, end_angle, step):
        rad = angle / 180 * np.pi
        if torch.is_tensor(x):
            aug_x = beamforming_torch(x, rad, beam_norm) # beamforming angle
        else:
            aug_x = beamforming(x, rad)
        aug_x_list.append(aug_x)    
    return aug_x_list


def random_beamform_aug(x, start_angle=-30, end_angle=31, step=5, n=2, beam_norm=False):
    angle_interval = np.arange(start=start_angle, stop=end_angle, step=step)
    random_angles = np.random.choice(angle_interval, n, replace=False)
    random_rads = random_angles / 180 * np.pi
    aug_x_list = []
    for random_rad in random_rads:
        if torch.is_tensor(x):
            aug_x = beamforming_torch(x, random_rad, beam_norm)
        else:
            aug_x = beamforming(x, random_rad)
        aug_x_list.append(aug_x)
    aug_x_list.append(x[:, :, 0, ...])
    return aug_x_list


def beamscan(x, start_angle=-30, end_angle=31, step=5):
    angle_interval = np.arange(start=start_angle, stop=end_angle, step=step)
    rads = angle_interval / 180 * np.pi
    aug_x_list = []
    for rad in rads:
        if torch.is_tensor(x):
            aug_x = beamforming_torch(x, rad)
        else:
            aug_x = beamforming(x, rad)
        aug_x_list.append(aug_x)
    aug_x_list.append(x[:, :, 0, ...])
    return aug_x_list
