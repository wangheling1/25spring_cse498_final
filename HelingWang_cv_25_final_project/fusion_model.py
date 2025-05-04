# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:42:06 2025

@author: 81265
"""

import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import matplotlib.pyplot as plt
import h5py
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import os

class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #print('x',x.shape)
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #x_ft = torch.fft.rfft(x, 2, normalized=True, onesided=True)#old version
        x_ft = torch.fft.rfft2(x)#new version

        # Multiply relevant Fourier modes
        #print('x_ft',x_ft.shape)
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        #x = torch.fft.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=( x.size(-2), x.size(-1)))#old version
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))#new version
        return x

class Deepwise(nn.Module):
    def __init__(self, width):
        super(Deepwise,self).__init__()
        self.conv0 = nn.Conv2d(width, width, 1)#, groups = width)
        self.conv1 = nn.Conv2d(width, width, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(width, width, 1)#, groups = width)
        
        
        #self.act1 = h_swish()
        self.batch1 = nn.BatchNorm2d(width)
        self.batch2 = nn.BatchNorm2d(width)
        self.batch3 = nn.BatchNorm2d(width)
        
    def forward(self,x):
        x = self.conv0(x)
        #x = self.batch1(x)
        x = F.relu(x)
        #x = self.act1(x)
        #x = self.conv1(x)
        #x = F.relu6(x)
        x = self.conv2(x)
        #x = self.batch3(x)
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(1, self.width) # input channel is 3: (a(x, y), x, y)
        self.fc0_1 = nn.Linear(5, 24) # input channel is 3: (a(x, y), x, y)
        #self.fc0 = nn.Linear(1, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.dw0 = Deepwise(self.width)
        self.dw1 = Deepwise(self.width)
        self.dw2 = Deepwise(self.width)
        self.dw3 = Deepwise(self.width)
        
        #self.dw0 = nn.Conv2d(in_channels, out_channels, kernel_size)

        self.fc1 = nn.Linear(self.width, 256)#layers
        self.fc2 = nn.Linear(256, 1)
        


    def forward(self, x):
        batchsize = x.shape[0]
        #print(x.shape)
        size_x, size_y = x.shape[1], x.shape[2]
        #size_x = 24
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        #print("after fc", x.shape)
        x = x.permute(0, 3, 1, 2)
        #print(x.shape)
        #x = x.permute(0, 3, 2, 1)
        #x = self.fc0_1(x)
        #x = x.permute(0, 1, 3, 2)
        
        #print("after permute",x.shape)#[20,32,5,256][batch_size,width,size_x,size_y]
        x1 = self.conv0(x)
        #print("after conv0",x1.shape)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.dw0(x)
        #print("x3.shape",x3.shape)
        x = x1 + x3
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.dw1(x)
        x = x1 + x3
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.dw2(x)
        x = x1 + x3
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.dw3(x)
        x = x1 + x3
        #print("before fc, permute", x.shape)
        #x = x.flatten(start_dim = 1, end_dim = 2)
        x = x.permute(0, 2, 3, 1)
        #print("before fc1,",x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.relu(x)
        
       
        #print("after fc2", x.shape)
        return x#*2000

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes,  width)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()
        #return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
# 指定 `.mat` 文件路径
addr = r"E:\desktop\cse498\final_project\processed_data_part1.mat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modes = 3
width = 32#32
learning_rate = 0.001
epochs = 5#500
step_size = 100
gamma = 0.5
batch_size = 16

with h5py.File(addr, "r") as f:
    data = np.array(f["Data"])
    segment_bitmap = np.array(f["SegmentBitmap"])
    layers_vector = np.array(f["LayersVector"])
    
    
input_image = data  # (1556, 1000, 256)
input_layers = layers_vector[:, :5, :]  # First 5 layers (1556, 5, 256)

# Extract outputs
output_segmentation = segment_bitmap  # (1556, 1000, 256)
output_layers = layers_vector[:, 5:, :]  # Next 15 layers (1556, 15, 256)

# Convert to PyTorch tensors
input_image = torch.tensor(input_image, dtype=torch.float32).unsqueeze(1)  # (1556, 1, 1000, 256)
input_layers = torch.tensor(input_layers, dtype=torch.float32)  # (1556, 5, 256)
output_segmentation = torch.tensor(output_segmentation, dtype=torch.float32)  # (1556, 1000, 256)
output_layers = torch.tensor(output_layers, dtype=torch.float32)  # (1556, 15, 256)

# Create a combined dataset dictionary
dataset = {
    "image": input_image,  # (1556, 1, 1000, 256)
    "layers": input_layers,  # (1556, 5, 256)
    "segmentation": output_segmentation,  # (1556, 1000, 256)
    "thickness": output_layers  # (1556, 15, 256)
}


# Train-test split (80% train, 20% test)
train_img, test_img, train_layers, test_layers, train_seg, test_seg, train_thick, test_thick = train_test_split(
    dataset["image"], dataset["layers"], dataset["segmentation"], dataset["thickness"], 
    test_size=0.2, random_state=42
)

# Create new dictionaries for training & testing
train_data = {
    "image": train_img,
    "layers": train_layers,
    "segmentation": train_seg,
    "thickness": train_thick
}

test_data = {
    "image": test_img,
    "layers": test_layers,
    "segmentation": test_seg,
    "thickness": test_thick
}

print("Train set sizes:", {k: v.shape for k, v in train_data.items()})
print("Test set sizes:", {k: v.shape for k, v in test_data.items()})

class IceRadarDataset(Dataset):
    def __init__(self, data):
        self.image = data["image"]
        self.layers = data["layers"]
        self.segmentation = data["segmentation"]
        self.thickness = data["thickness"]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return {
            "image": self.image[idx],  # (1, 1000, 256)
            "layers": self.layers[idx],  # (5, 256)
            "segmentation": self.segmentation[idx],  # (1000, 256)
            "thickness": self.thickness[idx],  # (15, 256)
        }

# Create DataLoaders
train_dataset = IceRadarDataset(train_data)
test_dataset = IceRadarDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

model = Net2d(modes, width).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#myloss = torch.nn.MSELoss()  #3.60
myloss = WingLoss()   #3.41
segmentation_loss_fn = WingLoss()
thickness_loss_fn = WingLoss()

for ep in range(epochs):
    for batch in train_loader:  # Each batch is a dictionary
        # Extract inputs
        #batch_size = batch["image"].shape[0]
        image_input = batch["image"].to(device)  # (batch_size, 1, 1000, 256)
        #print(image_input.shape)
        layer_input = batch["layers"].to(device)  # (batch_size, 5, 256)

        # Extract outputs
        segmentation_target = batch["segmentation"].to(device)  # (batch_size, 1000, 256)
        thickness_target = batch["thickness"].to(device)  # (batch_size, 15, 256)

        # Forward pass
        #segmentation_pred, thickness_pred = model(image_input, layer_input)
        segmentation_pred = model(image_input)
        # Compute loss (example: MSE loss for thickness, CrossEntropy for segmentation)
        seg_loss = segmentation_loss_fn(segmentation_pred.view(batch_size,-1), segmentation_target.view(batch_size,-1)) #loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        #thick_loss = thickness_loss_fn(thickness_pred.view(batch_size,-1), thickness_target.view(batch_size,-1))

        total_loss = seg_loss #+ thick_loss  # Weighted sum if needed

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    print(total_loss)
