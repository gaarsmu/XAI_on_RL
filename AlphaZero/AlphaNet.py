import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,3,1,1)
        self.relu = nn.LeakyReLU(5e-2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class AlphaNet(nn.Module):
    def __init__(self, depth, board_size=19, device='cpu'):
        super(AlphaNet, self).__init__()
        self.depth = depth
        self.first_conv = ConvBlock(1,64)
        self.board_size = board_size

        conv_blocks = []
        for i in range(2,self.depth+1):
            conv_blocks.append(ConvBlock(64,64))
        self.conv_blocks = nn.Sequential(*conv_blocks)


        self.policy_head_con = nn.Conv2d(64,2,1,1,0)
        self.policy_head_fc = nn.Linear(2*self.board_size*self.board_size,
                                         self.board_size*self.board_size)

        self.value_head_con = nn.Conv2d(64,1,1,1,0)
        self.value_head_fc1 = nn.Linear(self.board_size*self.board_size, 256)
        self.value_head_relu = nn.LeakyReLU(5e-2)
        self.value_head_fc2 = nn.Linear(256, 1)
        self.value_head_tanh = nn.Tanh()

        self.device = device
        self.to(self.device)

    
    def forward(self, x):
        x = self.first_conv(x)
        tower_output = self.conv_blocks(x)

        x = self.policy_head_con(tower_output)
        x = torch.reshape(x, (x.shape[0], -1))
        policy_output = self.policy_head_fc(x)

        x = self.value_head_con(tower_output)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.value_head_fc1(x)
        x = self.value_head_relu(x)
        x = self.value_head_fc2(x)
        value_output = self.value_head_tanh(x)

        return policy_output, value_output

    def predict(self, board):
        inpt = torch.from_numpy(board).to(torch.float)        
        inpt = inpt.reshape(1,1,19,19)
        inpt = inpt.to(self.device)
        probs, v = self.forward(inpt)
        return probs.detach().cpu().numpy(), v.item()      

if __name__ == '__main__':
    tensor = torch.randn(1,1,19,19)
    net = AlphaNet(8)
    net.eval()
    policy_output, value_output = net(tensor)
    print(value_output)
    print(F.softmax(policy_output, dim = 1).max())
    print(torch.argmax(F.softmax(policy_output, dim = 1)).item())