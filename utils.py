import numpy as np

BASE_VECS =  np.array(((1,1), (-1, 1), (1,0), (0,1)))
shift_ranges = np.arange(-4,5)

def check_victory(board, coords, val):
    for bv in BASE_VECS:
        x_coords = []
        y_coords = []
        for shift in shift_ranges:
            x,y = coords + shift*bv
            if (x >= 0 and x < board.shape[0] and 
               y >= 0 and y < board.shape[0]):
                x_coords.append(x)
                y_coords.append(y)
        vict, ends = check_vector(board[x_coords, y_coords], val)
        if vict:
            return True, ( x_coords[ends[0]], y_coords[ends[0]],
                           x_coords[ends[1]], y_coords[ends[1]])
    return False, (None, None, None, None)



def check_vector(vector, val):
    if (vector==val).sum()<5:
        return False, (None, None)
    count  = 0
    start = -1
    for i in range(vector.shape[0]):
        ent = vector[i]
        if ent == val:
            count += 1
            if start == -1:
                start =  i
        else:
            count = 0
            start = -1
        if count == 5:
            return True, (start, i)
    return False, (None, None)
