import numpy as np
from utils import *
from env import TicTacToeEnv
import tkinter as tk
from time import sleep

def mouseClick(event):
    if not env.freeze:
        x, y = int(event.x)// 25, int(event.y)// 25
        if 0 <= x <= 30 and 0 <= y <= 30:
            env.step((x , y ))
    waiter = False

class GameHost():

    def __init__(self, p1, p2):
        self.fp_turn = False
        self.p1 = p1
        self.p2 = p2

    def turn():
        global player_turn_wait
        if self.fp_turn:
            
            self.p1.turn()
        else:
            self.p2.turn()

class HumanPlayer():

    def __init__(self):
        self.type = 'human'

    def turn(self):
        pass

if __name__== '__main__':
    env = TicTacToeEnv(render=True)
    env.canv.tag_bind('rect', '<Button-1>', mouseClick)

    #player1 = HumanPlayer()
    #player2 = HumanPlayer()

    #host = GameHost(player1, player2)

    env.window.mainloop()