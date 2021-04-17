import numpy as np
from utils import *

class MCTS():

    def __init__(self, net, args):
        self.net = net
        self.args = args

        self.Qsa = {} #Edges q values
        self.Nsa = {} #edges number of visits
        self.Ns = {} #number of times board was visited
        self.Ps = {} #probs

        #self.Vs = {} #valid moves available

    def getProbs(self, env, temp=1):

        for _ in range(self.args['num_sims']):
            env_copy = env.copy_env()
            self.treeSearch(env_copy)
        
        s = env.getBoardHash()
        counts = [ self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(env.size*env.size) ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros((env.size*env.size,))
            probs[bestA] = 1
            return probs
        else:
            counts_sum = float(sum(counts))
            probs = [x/counts_sum for x in counts]
            return probs


    def treeSearch(self, env):
        
        s =  env.getBoardHash()

        #leaf node case
        if s not in self.Ps:
            probs, v = self.net.predict(env.getPBoard())

            self.Ps[s] = probs * (env.board == 0).reshape(1, -1)
            sum_probs = np.sum(self.Ps[s])
            if sum_probs > 0:
                self.Ps[s] = (self.Ps[s]/sum_probs).reshape(-1,)
            else:
                self.Ps[s] = (env.board == 0).reshape(1, -1)
                self.Ps[s] = (self.Ps[s]/np.sum(self.Ps[s])).reshape(-1,)
            self.Ns[s] = 0
            return -v
        
        else:
            cur_best = -float('inf')
            best_act = -1
            valid_moves = env.getValidMoves()
            for a in valid_moves:
                an = env.size*a[0] + a[1]
                if (s,an) in self.Qsa:
                    u = self.Qsa[(s,an)] +\
                        self.args['c']*self.Ps[s][an]*np.sqrt(self.Ns[s])/(1+self.Nsa[(s,an)])
                else:
                    u = self.args['c']*self.Ps[s][an]*np.sqrt(self.Ns[s])
                if u > cur_best:
                    cur_best = u
                    best_act = a
            
            a = best_act
            an = env.size*a[0] + a[1]
            _, done, reward, _ = env.step((a[0], a[1]))
            if done:
                v = reward
            else:
                v = self.treeSearch(env)

            if (s,an) in self.Qsa:
                self.Qsa[(s,an)] = (self.Nsa[(s,an)]*self.Qsa[(s,an)]+v)/(self.Nsa[(s,an)]+1)
                self.Nsa[(s,an)] += 1
            else:
                self.Qsa[(s,an)] = v
                self.Nsa[(s,an)] = 1

            self.Ns[s] += 1
            return -v
