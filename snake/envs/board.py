
import numpy as np 
import matplotlib.pyplot as plt

BODY_COLOR = np.array([1,0,0], dtype=np.uint8)
HEAD_COLOR = np.array([255, 0, 0], dtype=np.uint8)
FOOD_COLOR = np.array([255,0, 0], dtype=np.uint8)
SPACE_COLOR = np.array([255,255,255], dtype=np.uint8)


class Board(object):

    def __init__(self, height, weight):
        self.board = np.empty((height, weight, 3), dtype=np.uint8)
        self.board[:, :, :] = SPACE_COLOR

    def fill_cell(self, vertex, cell_size, color):
        x, y = vertex
        self.board[x:x+cell_size, y:y+cell_size, :] = color

# board = Board(400, 400)
# board.fill_cell([(70, 70), (80, 80)], FOOD_COLOR)
# plt.imshow(board.board)
# plt.show()

# print('hello')
