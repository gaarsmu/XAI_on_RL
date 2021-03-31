import numpy as np
from utils import check_victory
import tkinter as tk

class TicTacToeEnv():

    def __init__(self, render = False, size = 19):
        self.size = size
        self.board = np.zeros((size,size), dtype=np.int16)

        self.done = False
        self.freeze = False
        self.render = render
        if self.render:
            self.window = tk.Tk()
            self.canv = tk.Canvas(self.window, width=self.size*25, height=self.size*25)
            self.canv.grid(row=1, column=0, columnspan=2)
            
            
            for i in range(self.size):
                for j in range(self.size):
                    self.canv.create_rectangle(i*25, j*25,
                                 (i+1)*25, (j+1)* 25,
                                  tag="rect",
                                  fill="burlywood", outline="black")
            self.window.update()
        self.fpt = True
        self.turns_count = 0
        #self.board[self.size//2,self.size//2] = 1
        #if self.render:
        #    self.putfig(self.size//2,self.size//2)
        #    self.window.update()
        #self.turns_count = 1
        #self.fpt = False #first player turn
            
    
    def step(self, action):
        x,y = action
        self.turns_count += 1
        #repeated move case
        if self.board[x,y] != 0:
            self.done = True
            self.freeze = True
            self.fpt = not self.fpt
            return self.board, -1, self.done, None

        val = 1 if self.fpt else -1
        self.board[x,y] = val

        if self.render:
            self.putfig(x,y)
            self.window.update()
        self.fpt = not self.fpt

        done, positions = self.check_victory(x,y,val)
        self.done = done
        if done and self.render:
            self.canv.create_line(positions[0] * 25 + 13, positions[1] * 25 + 13,
                         positions[2] * 25 + 13, positions[3] * 25 + 13,
                         width=4, fill='red')
            self.window.update()
        reward = 1 if self.done else 0

        #Game won case
        if done:
            self.freeze = True
            
            return self.board, reward, self.done, None
        
        #Full board_case
        if self.turns_count == self.size*self.size:
            self.freeze = True
            done = True
            self.done = done

        return self.board, reward, self.done, None


    def check_victory(self, x,y, val):
        finish, positions = check_victory(self.board, (x,y), val)
        return finish, positions

    def putfig(self,x, y):
        if self.fpt:
            self.canv.create_oval(x * 25 + 1, y * 25 + 1,
                            x * 25 + 24, y * 25 + 24,
                            width=2, outline='black', fill='black')
        else:
            self.canv.create_oval(x * 25 + 1, y * 25 + 1,
                            x * 25 + 24, y * 25 + 24,
                            width=2, outline='white', fill='white')
        self.window.update()


