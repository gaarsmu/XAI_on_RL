import numpy as np
from utils import *
import time

class CalcBot():

    def __init__(self, depth, view_range=3, env_size=19):
        self.type = 'computer'
        self.depth = depth
        self.view_range = view_range
        self.env_size = env_size

    def get_action(self, board, info):
        move_scores = self.evaluate(board, info, self.depth, flip = 1)
        max_score = np.max([x[1] for x in move_scores])
        choices = [x[0] for x in move_scores if x[1] == max_score]
        choice_ind = np.random.choice(len(choices), 1)[0]
        out = choices[choice_ind]
        return out[0]+info['up'], out[1]+info['left']

    def evaluate(self, board, info, depth, flip):
        board_slice = board[info['up']:info['down']+1,
                            info['left']: info['right']+1]
        choices = np.transpose(np.nonzero(board_slice == 0))
        move_scores = []
        for choice in choices:
            x = choice[0]
            y = choice[1]
            board_loc = board.copy()
            board_loc[x+info['up'], y+info['left']] = flip
            victory, _ = check_victory(board_loc, (x+info['up'], y+info['left']),flip)
            if victory:
                move_scores.append((choice, flip))
                continue
            else:
                if depth == 1:
                    move_scores.append((choice, 0))
                    continue
                else:
                    info_loc = self.get_loc_info(info, x+info['up'], y+info['left'])
                    move_scores_loc = self.evaluate(board_loc, info_loc, depth-1, -1*flip)
                    if flip == 1:
                        move_scores.append( (choice, np.min([x[1] for x in move_scores_loc])) )
                    else:
                        move_scores.append( (choice, np.max([x[1] for x in move_scores_loc])) )
        return move_scores

    def get_loc_info(self, info_dict, x, y):
        info_dict_loc = {}
        info_dict_loc['up'] = min(info_dict['up'], max(0, x-self.view_range))
        info_dict_loc['down'] = max(info_dict['down'], min(self.env_size-1, x+self.view_range))
        info_dict_loc['left'] = min(info_dict['left'], max(0, y-self.view_range))
        info_dict_loc['right'] = max(info_dict['right'], min(self.env_size-1, y+self.view_range))
        return info_dict_loc

        #board_slice = board[info['up']:info['down'],
        #                    info['left']: info['right']]
        #choices = np.transpose(np.nonzero(board_slice == 0))
        #choice = np.random.choice(choices.shape[0], size=1)
        #out = choices[choice[0], :]