import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiscriminatorConv(nn.Module):

    def __init__(self, in_shape, hidden_dim, output_size, dropout_rate, slope):
        super(DiscriminatorConv, self).__init__()
        self._l1 = nn.Conv2d(in_shape[0], hidden_dim, 4, 2, 1, bias=False)
        self._b1 = nn.BatchNorm2d(hidden_dim)

        self._l2 = nn.Conv2d(hidden_dim, hidden_dim*2, 4, 2, 1, bias=False)
        self._b2 = nn.BatchNorm2d(hidden_dim*2)
        self._l3 = nn.Conv2d(hidden_dim*2, hidden_dim*3, 4, 2, 1, bias=False)
        self._b3 = nn.BatchNorm2d(hidden_dim*3)
        self._l4 = nn.Conv2d(hidden_dim*3, hidden_dim*4, 4, 2, 1, bias=False)
        self._b4 = nn.BatchNorm2d(hidden_dim*4)
        self._l5 = nn.Conv2d(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False)
        self._b5 = nn.BatchNorm2d(hidden_dim*8)
        self._l6 = nn.Conv2d(hidden_dim*8, 1, 4, 2, 1, bias=False)
        self._leaky_relu_slope = slope
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        z = self._l1(input)
        y = F.leaky_relu(self._b1(z), self._leaky_relu_slope)
        y = F.leaky_relu(self._b2(self._l2(y)), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self._b3(self._l3(y)), self._leaky_relu_slope)
        y = F.leaky_relu(self._b4(self._l4(y)), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self._b5(self._l5(y)), self._leaky_relu_slope)
        out = self._l6(y)
        return out


class DiscriminatorLin(nn.Module):

    def __init__(self, in_shape, hidden_dim, output_size, dropout_rate, slope):
        super(DiscriminatorLin, self).__init__()
        in_size = np.prod(in_shape)
        # Input and hidden linear layers.
        self.fc1 = nn.Linear(in_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        # Final output layer.
        self.fc4 = nn.Linear(hidden_dim, output_size)

        # Dropout layer for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu_slope = slope

    def forward(self, input):

        # Flatten the image.
        feature_nr = np.product(input[0].shape, axis=0)
        flatted = input.contiguous().view(-1, feature_nr)

        y = F.leaky_relu(self.fc1(flatted), self.leaky_relu_slope)
        y = self.dropout(y)
        y = F.leaky_relu(self.fc2(y), self.leaky_relu_slope)
        y = self.dropout(y)
        y = F.leaky_relu(self.fc3(y), self.leaky_relu_slope)
        y = self.dropout(y)

        out = nn.Sigmoid()(self.fc4(y))
        return out

class GeneratorMixed(nn.Module):

    def __init__(self, in_size, ch, output_shape, dropout_rate, slope):
        super(GeneratorMixed, self).__init__()
        self._image_shape = output_shape
        output_size = np.prod(output_shape)

        # convolution layer + Normalization
        # kernel_size=3, stride=1 and padding=1 keeps the image-size constant (1 per dim)
        # only channel-size will be changed in convolution layers
        self.l1 = nn.ConvTranspose2d(in_channels=in_size, out_channels=( ch*8), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.b1 = nn.BatchNorm2d(num_features=ch*8)

        self.l2 = nn.ConvTranspose2d(in_channels=ch*8, out_channels=ch*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),  bias=False)
        self.b2 = nn.BatchNorm2d(num_features=ch*4)

        self.l3 = nn.ConvTranspose2d(in_channels=ch*4, out_channels=ch*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),  bias=False)
        self.b3 = nn.BatchNorm2d(num_features=ch*2)

        self.l4 = nn.ConvTranspose2d(in_channels=ch*2, out_channels=ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),  bias=False)
        self.b4 = nn.BatchNorm2d(num_features=ch)
        
        # fully connected layer 
        self.fc1 = nn.Linear(ch, ch*2)
        self.fc2 = nn.Linear(ch*2,ch*4)
        self.fc3 = nn.Linear(ch*4, output_size)

    #     # Dropout layer for regularization.
        self._dropout = nn.Dropout(dropout_rate)
        self._leaky_relu_slope = slope

    def forward(self, x):
        
        # Leaky relu to help with gradient flow.
        y = F.leaky_relu(self.b1(self.l1(x)), self._leaky_relu_slope)
        y = F.leaky_relu(self.b2(self.l2(y)), self._leaky_relu_slope)
        y = F.leaky_relu(self.b3(self.l3(y)), self._leaky_relu_slope)
        y = F.leaky_relu(self.b4(self.l4(y)), self._leaky_relu_slope)
        
        # eliminating dimension of size 1
        # tensor-shape (<batchsize>,<ch>,1,1)->(<batchsize>,<ch>)
        z = torch.squeeze_copy(y)  

        y = F.leaky_relu(self.fc1(z), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self.fc2(y), self._leaky_relu_slope)
        y = self._dropout(y)

        # Final layer with tanh activation to be in range [-1,1].
        y = torch.tanh(self.fc3(y))
        
        shape = tuple([len(x)]+list(self._image_shape))
        return torch.reshape(y, shape)

    def getInputsize(self):
        return self.l1.in_channels 

class GeneratorMixed2(nn.Module):

    def __init__(self, in_size, ch, output_shape, dropout_rate, slope):
        super(GeneratorMixed2, self).__init__()
        self._image_shape = output_shape
        output_size = np.prod(output_shape)

        # convolution layer + Normalization
        # kernel_size=3, stride=1 and padding=1 keeps the image-size constant (1 per dim)
        # only channel-size will be changed in convolution layers
        self.l1 = nn.ConvTranspose2d(in_channels=in_size, out_channels=( ch*8), kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False)
        self.b1 = nn.BatchNorm2d(num_features=ch*8)

        self.l2 = nn.ConvTranspose2d(in_channels=ch*8, out_channels=ch*4, kernel_size=(4,4), stride=(2, 2), padding=(1, 1),  bias=False)
        self.b2 = nn.BatchNorm2d(num_features=ch*4)

        self.l3 = nn.ConvTranspose2d(in_channels=ch*4, out_channels=ch*2, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4),  bias=False)
        self.b3 = nn.BatchNorm2d(num_features=ch*2)

        self.l4 = nn.ConvTranspose2d(in_channels=ch*2, out_channels=ch, kernel_size=(3, 3), stride=(2,2), padding=(2, 2),  bias=False)
        self.b4 = nn.BatchNorm2d(num_features=ch)
        
        # fully connected layer 
        self.fc1 = nn.Linear(ch, ch*2)
        self.fc2 = nn.Linear(ch*2,ch*4)
        self.fc3 = nn.Linear(ch*4, output_size)

    #     # Dropout layer for regularization.
        self._dropout = nn.Dropout(dropout_rate)
        self._leaky_relu_slope = slope

    def forward(self, x):
        
        # Leaky relu to help with gradient flow.
        y = F.leaky_relu(self.b1(self.l1(x)), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self.b2(self.l2(y)), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self.b3(self.l3(y)), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self.b4(self.l4(y)), self._leaky_relu_slope)
        
        # eliminating dimension of size 1
        # tensor-shape (<batchsize>,<ch>,1,1)->(<batchsize>,<ch>)
        z = torch.squeeze_copy(y)  

        y = F.leaky_relu(self.fc1(z), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self.fc2(y), self._leaky_relu_slope)
        y = self._dropout(y)

        # Final layer with tanh activation to be in range [-1,1].
        y = torch.tanh(self.fc3(y))
        
        shape = tuple([len(x)]+list(self._image_shape))
        return torch.reshape(y, shape)

    def getInputsize(self):
        return self.l1.in_channels 

class GeneratorLin(nn.Module):

    def __init__(self, in_size, ch, output_shape, dropout_rate, slope):
        super(GeneratorLin, self).__init__()
        self._image_shape = output_shape

        output_size = np.prod(output_shape)
        self._fc1 = nn.Linear(in_size, ch*4)
        self._fc2 = nn.Linear(ch*4, ch*8)
        self._fc3 = nn.Linear(ch*8, ch*4)
        self._fc4 = nn.Linear(ch*4, output_size)

        # Dropout layer for regularization.
        self._dropout = nn.Dropout(dropout_rate)
        self._leaky_relu_slope = slope

    def forward(self, input):
        z = torch.squeeze_copy(input)
        y = F.leaky_relu(self._fc1(z), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self._fc2(y), self._leaky_relu_slope)
        y = self._dropout(y)
        y = F.leaky_relu(self._fc3(y), self._leaky_relu_slope)
        y = self._dropout(y)
        y = torch.tanh(self._fc4(y))

        shape = tuple([len(input)]+list(self._image_shape))

        return torch.reshape(y, shape)

    def getInputsize(self):
        return self._fc1.in_features

class GeneratorConv(nn.Module):

    def __init__(self, in_size, ch, output_shape, dropout_rate, slope):
        super(GeneratorConv, self).__init__()
        # Convolutionlayers with normalization
        self._image_shape = output_shape
        self.l1 = nn.ConvTranspose2d(
        in_channels=in_size, out_channels=ch*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=ch*8)
        self.l2 = nn.ConvTranspose2d( in_channels=ch*8, out_channels=ch*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=ch*4)
        self.l3 = nn.ConvTranspose2d( in_channels=ch*4, out_channels=ch*3, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=ch*3)
        self.l4 = nn.ConvTranspose2d(in_channels=ch*3, out_channels=ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=ch)
        self.l5 = nn.ConvTranspose2d( in_channels=ch, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=3)
        self.l6 = nn.ConvTranspose2d( in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)

        # Dropout layer for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu_slope = slope

    def forward(self, x):
        # Leaky relu to help with gradient flow.
        y = F.leaky_relu(self.bn1(self.l1(x)), self.leaky_relu_slope)
        y = self.dropout(y)
        y = F.leaky_relu(self.bn2(self.l2(y)), self.leaky_relu_slope)
        y = self.dropout(y)
        y = F.leaky_relu(self.bn3(self.l3(y)), self.leaky_relu_slope)
        y = self.dropout(y)
        y = F.leaky_relu(self.bn4(self.l4(y)), self.leaky_relu_slope)
        y = self.dropout(y)
        y = F.leaky_relu(self.bn5(self.l5(y)), self.leaky_relu_slope)
        y = self.dropout(y)
        y = F.leaky_relu(self.bn5(self.l6(y)), self.leaky_relu_slope)
        y = self.dropout(y)
        out = torch.tanh(self.l6(y))
        return out

    def getInputsize(self):
        return self.l1.in_channels

class GeneratorDC(nn.Module):
    def __init__(self, in_size, ch, output_shape, dropout_rate, slope):
        super(GeneratorDC, self).__init__()
        self.inputsize = in_size
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_size, ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.ConvTranspose2d(ch , ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)
        
    def getInputsize(self):
        return self.inputsize

class DiscriminatorDC(nn.Module):
    def __init__(self, in_shape, hidden_dim, output_size, dropout_rate, slope):
        super(DiscriminatorDC, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_shape[0], hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(dropout_rate, inplace=True),
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Dropout2d(dropout_rate),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)