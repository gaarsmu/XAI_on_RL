from env import TicTacToeEnv
from DQN_bot import DQN_bot
from DQN_net import DQNNet
import torch
import torch.nn as nn
from collections import deque
import random
import numpy as np
import time
torch.set_printoptions(sci_mode=False)

def get_trajectory(net, device, epsilon):
    env = TicTacToeEnv()
    player1 = DQN_bot(net=net, epsilon=epsilon, device=device)
    player2 = DQN_bot(net=net, epsilon=epsilon, device=device)

    done = False
    board = env.board
    trajectory = []
    dummy_info = {}

    while not done:
        if env.fpt:
            action = player1.get_action(board, dummy_info)
        else:
            action = player2.get_action(board*-1, dummy_info)
        new_board, reward, done, _ = env.step(action)
        if env.fpt:
            trajectory.append([board, action, reward, (-1)*new_board])
        else:
            trajectory.append([(-1)*board, action, reward, new_board])
        board = new_board
    if trajectory[-1][-2] == 1:
        trajectory[-2][-2] = -1
    return trajectory

def form_tensors(batch, batch_size):
    boards = []
    next_boards = []
    rewards = []
    actions = []
    masks = []
    for info in batch:
        board = info[0]
        a = info[1]
        reward = info[2]
        next_board = info[3]

        boards.extend([board, np.flip(board, 0),
         np.flip(board, 1), np.flip(board, (0,1))])
        actions.extend((a[0]*19+a[1], (18-a[0])*19 + a[1],
        a[0]*19 + (18-a[1]), (18-a[0])*19 + (18-a[1]) ))
        rewards.extend((reward,reward,reward,reward ))
        mask = 1 if reward==0 else 0
        masks.extend((mask, mask, mask, mask))
        next_boards.extend([next_board, np.flip(next_board, 0),
         np.flip(next_board, 1), np.flip(next_board, (0,1))])

    boards = np.array(boards).reshape(batch_size*4,1,19,19)
    rewards = np.array(rewards).reshape(batch_size*4)
    masks = np.array(masks).reshape(batch_size*4)
    next_boards = np.array(next_boards).reshape(batch_size*4,1,19,19)

    boards = torch.from_numpy(boards).type(torch.float).to(device)
    rewards = torch.from_numpy(rewards).type(torch.float).to(device)
    masks = torch.from_numpy(masks).type(torch.float).to(device)
    next_boards = torch.from_numpy(next_boards).type(torch.float).to(device)

    output = main_net(boards)
    target = torch.clone(output)
    target[(boards != 0).reshape(batch_size*4, -1)] = -1.
    next_values = main_net(next_boards)
    target[range(batch_size*4), actions] = rewards + gamma*masks*(-1*torch.max(next_values, dim=1)[0])
    target = target.detach()
    return output, target

def net_update():
    optimizer.zero_grad()
    batch_size = min(len(memory), BATCH_SIZE-len(memory_lm))
    batch = random.sample(memory, batch_size) + list(memory_lm)
    batch_size = batch_size + len(memory_lm)
    for sample in batch:
        output, target = form_tensors([sample], 1)
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    main_net = DQNNet(5)
    #main_net.load_state_dict(torch.load('DQN_net.pth'))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    main_net.to(device)
    memory = deque(maxlen=5000)
    memory_lm = deque(maxlen=8)
    n_epochs = 10000
    BATCH_SIZE = 32
    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(main_net.parameters(), lr=0.01)
    gamma = 0.95
    epsilons = [0.0, 1.0, 1.0, 1.0]
    start = time.time()

    for i in range(1, n_epochs+1):
        eps = epsilons[i%4] 
        traj = get_trajectory(main_net, device, eps)
        memory.extend(traj)
        memory_lm.extend(traj[-1:])
        net_update()
        if i % 100 == 0:
            torch.save(main_net.state_dict(), 'DQN_net.pth')
            print('Saving the model after {} episodes'.format(i))
        if i % 4 == 0:
            epsilons[0] = max(epsilons[1]*0.999, 0.05)
            epsilons[1] = max(epsilons[1]*0.999, 0.1)
            epsilons[2] = max(epsilons[2]*0.999, 0.25)
            epsilons[3] = max(epsilons[3]*0.999, 0.5)
        print('Episode: ', i, ', trajectory length: ', len(traj))
    finish = time.time()
    print(finish-start)
    