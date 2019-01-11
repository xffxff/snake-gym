
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from collections import deque
from gym.envs.classic_control import rendering

class SnakeAction(object):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class SnakeCellState(object):
    EMPTY = 0
    WALL = 1
    BITE = 2
    FOOD = 3

class SnakeReward(object):
    ALIVE = 0.
    FOOD = 1.
    DEAD = -1.
    WON = 100.

class BoardColor(object):
    BODY_COLOR = np.array([0, 0, 0], dtype=np.uint8)
    HEAD_COLOR = np.array([255, 0, 0], dtype=np.uint8)
    FOOD_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    SPACE_COLOR = np.array([255,255,255], dtype=np.uint8)


class SnakeEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 50
    }

    def __init__(self):
        self.snake = deque()
        self.width = 20
        self.hight = 20
        self.snake_start = (10, 10)

        self.action_space = spaces.Discrete(4)

        self.snake = deque()
        self.snake_head = None
        self.empty_cells = None
        self.prev_action = None
        self.food = None
        self.viewer = None

    def reset(self):
        self.prev_action = SnakeAction.UP
        self.snake_head = self.snake_start
        self.empty_cells = [(x, y) for x in range(self.width) for y in range(self.hight)]
        self.snake.clear()
        self.snake.append(self.snake_start)
        self.empty_cells.remove(self.snake_start)
        self.food = self.empty_cells[self.np_random.choice(len(self.empty_cells))]
        self.empty_cells.remove(self.food)
        return self.get_image()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if not self.is_valid_action(action):
            action = self.prev_action
        self.prev_action = action

        next_head = self.next_head(action)
        next_head_state = self.cell_state(next_head)

        self.snake_head = next_head
        self.snake.appendleft(next_head)

        done = False
        if next_head_state == SnakeCellState.WALL:
            reward = SnakeReward.DEAD
            done = True
        elif next_head_state == SnakeCellState.BITE:
            reward = SnakeReward.DEAD
            done = True
        elif next_head_state == SnakeCellState.FOOD:
            if len(self.empty_cells) > 0:
                self.food = self.empty_cells[self.np_random.choice(len(self.empty_cells))]
                self.empty_cells.remove(self.food)
                reward = SnakeReward.FOOD
            else:
                reward = SnakeReward.WON
                done = True
        else:
            self.empty_cells.remove(self.snake_head)
            emtpy_cell = self.snake.pop()
            self.empty_cells.append(emtpy_cell)
            reward = 0.
        return self.get_image(), reward, done, {}

    def get_image(self):
        board_width = 400
        board_height = 400
        cell_size = int(board_width / self.width)

        board = Board(board_height, board_width)
        for x, y in self.snake:
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.BODY_COLOR)
        x, y = self.snake[0]
        board.fill_cell((x*cell_size, y*cell_size), cell_size, color=BoardColor.HEAD_COLOR)
        
        if self.food:
            x, y = self.food
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.FOOD_COLOR)
        return board.board

    def next_head(self, action):
        x, y = self.snake_head
        if action == SnakeAction.LEFT:
            return (x, y - 1)
        if action == SnakeAction.RIGHT:
            return (x, y + 1)
        if action == SnakeAction.UP:
            return (x - 1, y)
        return (x + 1, y)

    def cell_state(self, cell):
        if cell in self.empty_cells:
            return SnakeCellState.EMPTY
        if cell == self.food:
            return SnakeCellState.FOOD
        if cell in self.snake:
            return SnakeCellState.BITE
        return SnakeCellState.WALL

    def is_valid_action(self, action):
        if len(self.snake) == 1:
            return True
        
        horizontal_actions = [SnakeAction.LEFT, SnakeAction.RIGHT]
        vertical_actions = [SnakeAction.UP, SnakeAction.DOWN]

        if self.prev_action in horizontal_actions:
            return action in vertical_actions
        return action in horizontal_actions

    def render(self, mode='human'):
        img = self.get_image()
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        return self.viewer.isopen


class Board(object):

    def __init__(self, height, weight):
        self.board = np.empty((height, weight, 3), dtype=np.uint8)
        self.board[:, :, :] = BoardColor.SPACE_COLOR

    def fill_cell(self, vertex, cell_size, color):
        x, y = vertex
        self.board[x:x+cell_size, y:y+cell_size, :] = color