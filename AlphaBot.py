import numpy as np
from utils import getBoardSims
from AlphaZero.MCTS import MCTS
from tqdm import tqdm
from AlphaZero.AlphaNet import AlphaNet

class AlphaBot():

    def __init__(self, path, args):
        self.type = 'computer'
        self.net = AlphaNet(11, device='cuda:0')
        self.net.load_net(path)
        self.mcts = MCTS(self.net, args)


    def get_action(self, env, info):
        probs = self.mcts.getProbs(env, temp=0)
        action = np.random.choice(probs.shape[0], p=probs.reshape(-1,))
        return action//env.size, action % env.size
