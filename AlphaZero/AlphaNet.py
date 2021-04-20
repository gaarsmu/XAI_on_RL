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

    def __init__(self, depth, board_size=19, device='cpu', lr = 0.001):
        super(AlphaNet, self).__init__()
        self.depth = depth
        self.first_conv = ConvBlock(1,64)
        self.board_size = board_size
        self.lr = lr

        #NNet blocks
        conv_blocks = []
        for _ in range(2,self.depth+1):
            conv_blocks.append(ConvBlock(64,64))
        self.conv_blocks = nn.Sequential(*conv_blocks)


        self.policy_head_con = nn.Conv2d(64,2,1,1,0)
        self.policy_head_fc = nn.Linear(2*self.board_size*self.board_size,
                                         self.board_size*self.board_size)
        self.policy_head_softmax = nn.Softmax(dim=1)

        self.value_head_con = nn.Conv2d(64,1,1,1,0)
        self.value_head_fc1 = nn.Linear(self.board_size*self.board_size, 256)
        self.value_head_relu = nn.LeakyReLU(5e-2)
        self.value_head_fc2 = nn.Linear(256, 1)
        self.value_head_tanh = nn.Tanh()

        #Loss and optimizer
        self.policy_loss = cross_entropy
        self.value_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.device = device
        self.to(self.device)

    
    def forward(self, x):
        x = self.first_conv(x)
        tower_output = self.conv_blocks(x)

        x = self.policy_head_con(tower_output)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.policy_head_fc(x)
        policy_output = self.policy_head_softmax(x)

        x = self.value_head_con(tower_output)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.value_head_fc1(x)
        x = self.value_head_relu(x)
        x = self.value_head_fc2(x)
        value_output = self.value_head_tanh(x)

        return policy_output, value_output

    def predict(self, board):
        with torch.no_grad():
            inpt = torch.from_numpy(board).to(torch.float)        
            inpt = inpt.reshape(1,1,19,19)
            inpt = inpt.to(self.device)
            probs, v = self.forward(inpt)
            return probs.detach().cpu().numpy(), v.item()

    def updateNet(self, data, epochs = 10, batch_size=64):
        for _ in range(epochs):
            indexes = list(np.random.permutation(range(len(data))))
            for i in range(0, len(data), batch_size):
                loc_indexes = indexes[i:i+batch_size]
                batch = data[loc_indexes]
                batch_len = len(batch)
                boards = [x[0] for x in batch]
                boards = np.array(boards).reshape(batch_len,1,19,19)
                boards = torch.from_numpy(boards).type(torch.float).to(self.device)

                probs = [x[1] for x in batch]
                probs = np.array(probs).reshape(batch_len, 19*19)
                probs = torch.from_numpy(probs).type(torch.float).to(self.device)

                values = [x[2] for x in batch]
                values = np.array(values).reshape(batch_len, 1)
                values = torch.from_numpy(values).type(torch.float).to(self.device)

                policy_out, values_out = self.forward(boards)
                loss1 = self.policy_loss(probs, policy_out)
                loss2 = self.value_loss(values, values_out)
                loss = loss1 + loss2
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def save_net(self, path):
        torch.save(self.state_dict(), path)

    def load_net(self, path):
        self.load_state_dict(torch.load(path))


def cross_entropy(target, output):
    return torch.mean(-target*torch.log(output))

if __name__ == '__main__':
    tensor = torch.randn(1,1,19,19)
    net = AlphaNet(8)
    net.eval()
    policy_output, value_output = net(tensor)
    print(value_output)
    print(F.softmax(policy_output, dim = 1).max())
    print(torch.argmax(F.softmax(policy_output, dim = 1)).item())