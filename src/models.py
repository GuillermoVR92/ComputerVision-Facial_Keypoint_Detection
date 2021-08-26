## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # INPUT = 224x224x1
        
        output_size = 136
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        # Output = (W-kernel)/Stride + 1 = (224 - 4)/1 + 1 = 220
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        # Output = (W-kernel)/Stride + 1 = (220 - 2)/2 + 1 = 110
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batch1 = nn.BatchNorm2d(32)
        
        # Output = (W-kernel)/Stride + 1 = (110 - 3)/1 + 1 = 107
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Output = (W-kernel)/Stride + 1 = (107 - 2)/2 + 1 = 53
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batch2 = nn.BatchNorm2d(64)
        
        # Output = (W-kernel)/Stride + 1 = (53 - 2)/1 + 1 = 51
        self.conv3 = nn.Conv2d(64, 128, 2)
        # Output = (W-kernel)/Stride + 1 = (51 - 2)/2 + 1 = 25
        self.pool3 = nn.MaxPool2d(2, 2)
        self.batch3 = nn.BatchNorm2d(128)
        
        # Output = (W-kernel)/Stride + 1 = (25 - 1)/1 + 1 = 24
        self.conv4 = nn.Conv2d(128, 256, 1)
        # Output = (W-kernel)/Stride + 1 = (23 - 2)/2 + 1 = 13
        self.pool4 = nn.MaxPool2d(2, 2)
        self.batch4 = nn.BatchNorm2d(256)
        
        #Output = 13x13x256
                
        #self.dropout1= nn.Dropout2d(p=0.1, inplace=False)
        #self.dropout2= nn.Dropout2d(p=0.2, inplace=False)
        #self.dropout3= nn.Dropout2d(p=0.3, inplace=False)
        #self.dropout4= nn.Dropout2d(p=0.4, inplace=False)
        self.dropout5= nn.Dropout(p=0.5)
        self.dropout6= nn.Dropout(p=0.6)
             
        self.fc1 = nn.Linear(in_features=13*13*256, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000) 
        self.fc3 = nn.Linear(in_features=1000, out_features=output_size) 
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        #x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        #x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        #x = self.dropout4(self.pool4(F.relu(self.conv4(x))))
        x = self.batch1(self.pool1(F.relu(self.conv1(x))))
        x = self.batch2(self.pool2(F.relu(self.conv2(x))))
        x = self.batch3(self.pool3(F.relu(self.conv3(x))))
        x = self.batch4(self.pool4(F.relu(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
