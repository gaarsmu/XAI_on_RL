import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from env import TicTacToeEnv
from utils import *
from AlphaZero.AlphaNet import AlphaNet
from AlphaZero.MCTS import MCTS
from AlphaZero.trainAlphaZero import Trainer
import torch

if __name__ == '__main__':
    net = AlphaNet(11, device='cuda:0')
    net.load_net('AlphaZero/models/net_updates_1.pth')
    args = {'c': 1., 'num_sims': 25, 'sleep_time': 0}
    mcts = MCTS(net, args)
    coach = Trainer(net, args, num_eps=100)

    for num_update in range(501, 1001):
        print('Starting update {}'.format(num_update))
        coach.execute_update(net, args)

        net_old = AlphaNet(11, device='cuda:0')
        net_old.load_net('AlphaZero/models/net_updates_'+str(num_update-1)+'.pth')

        results = coach.arena(net, net_old, args, games_to_play=40)

        num_of_wins = np.sum([1 if results[i] ==(-1)**(i) else 0 for i in range(len(results))])
        num_of_losses = np.sum([1 if results[i] ==(-1)**(i+1) else 0 for i in range(len(results))])
        num_of_draws = np.sum([1 if results[i] == 0 else 0 for i in range(len(results))])

        print('Update {} finished'.format(num_update))
        print('New agent wins {}'.format(num_of_wins))
        print('New agent draws {}'.format(num_of_draws))
        print('New agent losses {}'.format(num_of_losses))

        des_result = np.array([(-1)**i * results[i] for i in range(len(results))])
        if des_result.sum() < 1:
            print('Update worse than 1, keeping previous version')
            net_old.save_net('AlphaZero/models/net_updates_'+str(num_update)+'.pth')
            net.load_net('AlphaZero/models/net_updates_'+str(num_update)+'.pth')
        else:
            net.save_net('AlphaZero/models/net_updates_'+str(num_update)+'.pth')


