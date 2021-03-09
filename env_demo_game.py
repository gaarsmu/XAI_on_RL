import numpy as np
from env import TicTacToeEnv
from time import sleep

if __name__=='__main__':
    env = TicTacToeEnv(render=True)
    sleep(2)
    env.step((16,17))
    sleep(2)
    env.step((16,16))
    sleep(2)
    env.step((17,16))
    sleep(2)
    env.step((17,17))
    sleep(2)
    env.step((14,16))
    sleep(2)
    env.step((14,14))
    sleep(2)
    env.step((14,15))
    sleep(2)
    env.step((13,13))
    env.window.mainloop()