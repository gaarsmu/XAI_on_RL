import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,3,1,1)
        self.relu = nn.LeakyReLU(5e-2)
        #self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        #x = self.norm(x)
        return x

class DQNNet(nn.Module):
    def __init__(self, depth):
        super(DQNNet, self).__init__()
        self.depth = depth
        self.first_conv = ConvBlock(1,64)

        conv_blocks = []
        for i in range(2,self.depth+1):
            conv_blocks.append(ConvBlock(64,64))
        self.conv_blocks = nn.Sequential(*conv_blocks)


        self.head_con = nn.Conv2d(64,2,1,1,0)
        #self.head_norm = nn.BatchNorm2d(2)
        self.head_relu = nn.ReLU()
        self.head_fc = nn.Linear(2*19*19, 19*19)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.conv_blocks(x)
        x = self.head_con(x)
        #x = self.head_norm(x)
        x = self.head_relu(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.head_fc(x)
        return x

if __name__ == '__main__':
    tensor = torch.randn(1,1,19,19)
    net = DQNNet(11)
    net.eval()
    output = net(tensor)
    print(torch.argmax(output).item())