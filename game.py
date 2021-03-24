import numpy as np
from utils import *
from env import TicTacToeEnv
import tkinter as tk
from time import sleep

def mouseClick(event):
    global host
    if not env.freeze and not host.freeze and not host.game_over:
        
        x, y = int(event.x)// 25, int(event.y)// 25
        if 0 <= x <= 30 and 0 <= y <= 30:
            #print('clicked on ', x, ' ', y)
            host.freeze = True
            host.turn(x,y)
            
            if host.game_type == 'hc' and not host.game_over:
                x, y = host.p2.get_action(host.env.board, host.info_dict)
                host.turn(x,y)
                host.freeze = False
        else:
            pass
    else:
        pass

class GameHost():

    def __init__(self, p1, p2, env):
        self.fp_turn = False
        self.p1 = p1
        self.p2 = p2
        self.freeze = True
        self.game_over = False
        self.env = env
        self.info_dict = {'up' : 12, 'down' : 18, 'left' : 12, 'right' : 18}

    def start_game(self):
        #self.game_in_progress = True
        if self.p1.type == 'human' and self.p2.type == 'human':
            self.game_type = 'hh'
            self.env.canv.tag_bind('rect', '<Button-1>', mouseClick)
        elif self.p1.type == 'human' and self.p2.type == 'computer':
            self.game_type = 'hc'
            self.env.canv.tag_bind('rect', '<Button-1>', mouseClick)
        elif self.p1.type == 'computer' and self.p2.type == 'computer':
            self.game_type = 'cc'

        if self.game_type == 'hh':
            self.freeze = False
        elif self.game_type == 'hc':
            x,y = self.p2.get_action(host.env.board, host.info_dict)
            self.turn(x,y)
            self.freeze = False
        elif self.game_type == 'cc':
            while not self.game_over:
                if self.fp_turn:
                    x, y = self.p1.get_action(self.env.board, self.info_dict)
                else:
                    x, y = self.p2.get_action(self.env.board, self.info_dict)
                print(x, y, self.info_dict)
                self.turn(x,y)
        

    def turn(self, x, y):
        board, reward, done, info = env.step((x , y ))
        self.update_info(x, y)
        self.board = board
        if done:
            self.game_over = True
        self.fp_turn = not self.fp_turn

    def update_info(self, x, y):
        self.info_dict['up'] = min(self.info_dict['up'],max(0, x-3))
        self.info_dict['down'] = max(self.info_dict['down'] , min(30, x+3))
        self.info_dict['left'] = min(max(0, y-3), self.info_dict['left'])
        self.info_dict['right'] = max(self.info_dict['right'], min(30, y+3))

class HumanPlayer():

    def __init__(self):
        self.type = 'human'

    def get_action(self):
        pass

class RandomBot():

    def __init__(self):
        self.type = 'computer'

    def get_action(self, board, info):
        board_slice  = board[info['up']:info['down'], info['left']: info['right']]
        choices = np.transpose(np.nonzero(board_slice==0))
        choice = np.random.choice(choices.shape[0], size=1)
        out = choices[choice[0], :]
        return out[0]+info['up'], out[1]+info['left']

if __name__== '__main__':
    env = TicTacToeEnv(render=True)

    player1 = RandomBot()
    player2 = RandomBot()

    host = GameHost(player1, player2, env)
    host.start_game()
    env.window.mainloop()