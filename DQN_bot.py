import torch
from DQN_net import DQNNet
import numpy as np

class DQN_bot():
    def __init__(self, epsilon = 0, depth = 11, save_path=None):
        self.type = 'computer'
        self.net = DQNNet(depth)
        self.epsilon = epsilon
        if save_path is not None:
            self.net.load_state_dict(torch.load(save_path))
    
    def get_action(self, board, info):
        if np.random.uniform() < self.epsilon:
            choices = np.transpose(np.nonzero(board == 0))
            choice = np.random.choice(choices.shape[0], size=1)
            out = choices[choice[0], :]
            return out[0], out[1]         
        else:
            inpt = torch.from_numpy(board).to(torch.float)        
            inpt = inpt.reshape(1,1,19,19)
            values = self.net(inpt)
            choice = torch.argmax(values).item()
            return choice//19, choice % 19
