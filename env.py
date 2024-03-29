import numpy as np
from utils import check_victory
import tkinter as tk

class TicTacToeEnv():

    def __init__(self, render=False, size=19, gap=2):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int16)

        self.done = False
        self.freeze = False
        self.render = render
        self.g = gap
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
            self.vis_objects = []
            self.end_obj= None
        self.fpt = True
        self.turns_count = 0
        self.moves_log = []
        self.borders_log = []
    
    def step(self, action):
        x, y = action
        self.moves_log.append(action)
        #Update borders
        x_min = max(x-self.g, 0)
        x_max = min(x+self.g+1, 19)
        y_min = max(y-self.g, 0)
        y_max = min(y+self.g+1, 19)

        if self.turns_count == 0:
            self.borders_log.append((x_min, x_max, y_min, y_max))
        else:
            x1, x2, y1, y2 = self.borders_log[-1]
            self.borders_log.append((min(x_min, x1), max(x_max, x2), min(y_min, y1), max(y_max, y2)))
        self.turns_count += 1

        #repeated move case
        if self.board[x, y] != 0:
            self.done = True
            self.freeze = True
            self.fpt = not self.fpt
            if self.render:
                self.end_obj = self.canv.create_oval(x * 25 + 1, y * 25 + 1,
                            x * 25 + 24, y * 25 + 24,
                            width=2, outline='red', fill='red')
                self.window.update()
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
            self.end_obj = self.canv.create_line(positions[0] * 25 + 13, positions[1] * 25 + 13,
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
            self.done = True

        return self.board, reward, self.done, None

    def step_back(self):
        if self.turns_count > 0:
            x, y = self.moves_log[-1]
            self.moves_log = self.moves_log[:-1]
            self.borders_log = self.borders_log[:-1]
            self.turns_count -= 1
            self.board[x,y] = 0
            self.fpt = not self.fpt
            self.done = False
            self.freeze = False
            if self.render:
                self.canv.delete(self.vis_objects[-1])
                self.vis_objects = self.vis_objects[:-1]
                if self.end_obj is not None:
                    self.canv.delete(self.end_obj)
                    self.end_obj = None
                self.window.update()

    def check_victory(self, x,y, val):
        finish, positions = check_victory(self.board, (x,y), val)
        return finish, positions

    def putfig(self,x, y):
        if self.fpt:
            ov = self.canv.create_oval(x * 25 + 1, y * 25 + 1,
                            x * 25 + 24, y * 25 + 24,
                            width=2, outline='black', fill='black')
        else:
            ov = self.canv.create_oval(x * 25 + 1, y * 25 + 1,
                            x * 25 + 24, y * 25 + 24,
                            width=2, outline='white', fill='white')
        self.vis_objects.append(ov)
        self.window.update()

    def getBoardHash(self):
        return ''.join([str(x) for x in self.board.reshape(-1,) + 1])

    def getPBoard(self):
        return self.board if self.fpt else (-1)*self.board

    def getValidMoves(self):
        if self.turns_count == 0:
            if np.random.random() < 0.5:
                return np.array([[9, 9]])
            else:
                x = np.random.choice(range(7, 12))
                y = np.random.choice(range(7, 12))
                return np.array([[x, y]])
        else:
            x1, x2, y1, y2 = self.borders_log[-1]
            result = np.transpose(np.nonzero(self.board[x1:x2, y1:y2] == 0))
            result = result + np.array([[x1, y1]])
            return result

    def copy_env(self):
        new_env = TicTacToeEnv()

        new_env.size = self.size
        new_env.board = self.board.copy()

        new_env.done = self.done
        new_env.freeze = self.freeze
        new_env.fpt = self.fpt
        new_env.turns_count = self.turns_count
        return new_env   


