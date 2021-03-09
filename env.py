import numpy as np
from utils import check_victory
import tkinter as tk

class TicTacToeEnv():

    def __init__(self, render = False):
        self.board = np.zeros((31,31), dtype=np.int16)

        self.done = False
        self.freeze = False
        self.render = render
        if self.render:
            self.window = tk.Tk()
            self.canv = tk.Canvas(self.window, width=775, height=775)
            self.canv.grid(row=1, column=0, columnspan=2)
            
            
            for i in range(0, 31, ):
                for j in range(0, 31):
                    self.canv.create_rectangle(i*25, j*25,
                                 (i+1)*25, (j+1)* 25,
                                  tag="rect",
                                  fill="lightgray", outline="black")
            self.window.update()
        self.fpt = True
        self.board[15,15] = 1
        if self.render:
            self.putfig(15,15)
            self.window.update()
        self.fpt = False #first player turn
            
    
    def step(self, action):
        x,y = action
        
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
                         width=2, fill='blue')
            self.window.update()
        if done:
            self.freeze = True
        reward = 1 if self.done else 0
        return self.board, reward, self.done, None


    def check_victory(self, x,y, val):
        finish, positions = check_victory(self.board, (x,y), val)
        return finish, positions

    def putfig(self,x, y):
        if self.fpt:
            self.canv.create_oval(x * 25 + 1, y * 25 + 1,
                            x * 25 + 24, y * 25 + 24,
                            width=2, outline='black')
        else:
            self.canv.create_oval(x * 25 + 1, y * 25 + 1,
                            x * 25 + 24, y * 25 + 24,
                            width=2, outline='red')
        self.window.update()


