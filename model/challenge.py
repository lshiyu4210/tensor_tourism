"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Challenge(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        ## TODO: define your model architecture
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 8, 3, 1, 1)
        self.fc_1 = nn.Linear(in_features=32, out_features=2)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        ## TODO: initialize the parameters for your network
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]
        input_size = self.fc_1.weight.size(1)
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(input_size))
        nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape

        ## TODO: implement forward pass for your network
        x = self.conv1(x)                   # 3, 16
        x = nn.functional.relu(x)
        # print(x.shape, 1)
        x = self.pool(x)
        # print(x.shape, 1)
        
        x = self.conv2(x)                   # 16, 64
        x = nn.functional.relu(x)
        # print(x.shape, 2)
        x = self.pool(x)
        # print(x.shape, 2)
        
        x = self.conv3(x)                   # 64, 128
        x = nn.functional.relu(x)
        x = self.pool(x)
        # print(x.shape, 3)
        
        x = self.conv4(x)                   # 128, 128
        x = nn.functional.relu(x)
        x = self.pool(x)
        # print(x.shape, 4)
        
        x = self.conv5(x)                   # 128, 8
        x = nn.functional.relu(x)
        x = self.pool(x)
        # print(x.shape, 5)
        
        x = x.view(x.size(0), -1)
        # print(x.shape, 'fc')
        z = self.fc_1(x)
        # print(N, C, H, W, 'fc')

        return z
