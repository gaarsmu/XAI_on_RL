import numpy as np
from env import TicTacToeEnv
from utils import getBoardSims
from AlphaZero.MCTS import MCTS
from tqdm import tqdm

class Trainer():

    def __init__(self, agent, mcts_args,temp_thres = 16, num_eps = 20):
        self.temp_thres = temp_thres
        self.mcts = MCTS(agent, mcts_args)
        self.num_eps = num_eps 

    def play_game(self, agent):
        train_samples = []
        episode_step = 0
        env = TicTacToeEnv()

        while True:
            episode_step += 1
            temp = int(episode_step < self.temp_thres)
            probs = self.mcts.getProbs(env, temp=temp)

            action = np.random.choice(probs.shape[0], p=probs.reshape(-1,))
            #probs_orig, val = self.mcts.net.predict(env.getPBoard())
            for board_s, probs_s in getBoardSims(env.getPBoard(), probs):
                train_samples.append([board_s, probs_s, env.fpt]) #, probs_orig, action, val
            _, reward, done, _ = env.step( (action//env.size, action % env.size) )
            if done:
                final_player = env.fpt
                return [ (x[0], x[1], reward if final_player != x[2] else -reward) for x in train_samples] #, x[3], x[4], x[5]

    def execute_update(self, agent, mcts_args):
        train_samples = []
        for _ in range(self.num_eps): #tqdm()
            self.mcts = MCTS(agent, mcts_args)
            train_samples += self.play_game(agent)

        agent.updateNet(np.array(train_samples))
        return train_samples

    def arena(self, agent1, agent2, mcts_args, games_to_play=10):
        mcts1 = MCTS(agent1, mcts_args)
        mcts2 = MCTS(agent2, mcts_args)
        results = []

        for i in range(games_to_play): #tqdm()
            if i % 2 == 0:
                player1 = mcts1
                player2 = mcts2
            else:
                player2 = mcts1
                player1 = mcts2

            env = TicTacToeEnv()

            done = False
            while not done:
                first_player_move = env.fpt
                if first_player_move:
                    probs = player1.getProbs(env, temp=0)
                else:
                    probs = player2.getProbs(env, temp=0)
                
                action = np.random.choice(probs.shape[0], p=probs.reshape(-1,))
                _, reward, done, _ = env.step( (action//env.size, action % env.size) )
                if reward == -1:
                    print('Repeated move!')
                if done:
                    results.append( reward if first_player_move else -1*reward )
        return results



