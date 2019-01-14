

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import math
from collections import deque


class SnakeAction(object):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class BoardColor(object):
    BODY_COLOR = np.array([0, 0, 0], dtype=np.uint8)
    HEAD1_COLOR = np.array([255, 0, 0], dtype=np.uint8)
    HEAD2_COLOR = np.array([0, 0, 255], dtype=np.uint8)
    FOOD_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    SPACE_COLOR = np.array([255,255,255], dtype=np.uint8)


class SnakeEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 50
    }

    def __init__(self):
        self.width = 20
        self.hight = 20

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(400, 400, 3), dtype=np.uint8)

        self.snake = None
        self.prev_act = None
        self.food = None
        self.viewer = None
        self.np_random = np.random

    def reset(self):
        self.snake = Snake()
        empty_cells = self.get_empty_cells()
        empty_cells = self.snake.init(empty_cells, self.np_random)
        self.food = empty_cells[self.np_random.choice(len(empty_cells))]
        return self.get_image()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if not self.is_valid_action(action):
            action = self.prev_act
        self.prev_act = action

        snake_tail = self.snake.step(action)

        reward = 0.
        done = False
            
        if self.snake.head == self.food:
            reward += 1.
            self.snake.snake.append(snake_tail)
            empty_cells = self.get_empty_cells()
            self.food = empty_cells[self.np_random.choice(len(empty_cells))]
        
        #snake collided wall
        if self.is_collided_wall(self.snake.head):
            reward -= 1.
            done = True
        
        #snake bite itself 
        if self.snake.head in self.snake.body:
            reward -= 1.
            done = True

        return self.get_image(), reward, done, {}

    def get_image(self):
        board_width = 400
        board_height = 400
        cell_size = int(board_width / self.width)

        board = Board(board_height, board_width)
        for x, y in self.snake.body:
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.BODY_COLOR)

        x, y = self.snake.head
        board.fill_cell((x*cell_size, y*cell_size), cell_size, color=BoardColor.HEAD1_COLOR)

        if self.food:
            x, y = self.food
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.FOOD_COLOR)
        return board.board

    def get_empty_cells(self):
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.hight)]
        for cell in self.snake.snake:
            if cell in empty_cells:
                empty_cells.remove(cell)
        if self.food in empty_cells:
            empty_cells.remove(self.food)
        return empty_cells

    def is_valid_action(self, action):
        if len(self.snake.snake) == 1:
            return True
        
        horizontal_actions = [SnakeAction.LEFT, SnakeAction.RIGHT]
        vertical_actions = [SnakeAction.UP, SnakeAction.DOWN]

        if self.prev_act in horizontal_actions:
            return action in vertical_actions
        return action in horizontal_actions

    def is_collided_wall(self, head):
        x, y = head
        if x < 0 or x > 19 or y < 0 or y > 19:
            return True
        return False

    def render(self, mode='human'):
        img = self.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class Snake(object):

    def __init__(self):
        self.snake = deque()
        
    def step(self, action):
        x, y = self.head
        if action == SnakeAction.LEFT:
            self.snake.appendleft((x, y - 1))
        if action == SnakeAction.RIGHT:
            self.snake.appendleft((x, y + 1))
        if action == SnakeAction.UP:
            self.snake.appendleft((x - 1, y))
        if action == SnakeAction.DOWN:
            self.snake.appendleft((x + 1, y))
        return self.snake.pop()

    @property
    def head(self):
        return self.snake[0]

    @property
    def body(self):
        return list(self.snake)[1:]
    
    def init(self, empty_cells, np_random):
        start_head = empty_cells[np_random.choice(len(empty_cells))]
        self.snake.appendleft(start_head)
        empty_cells.remove(start_head)
        return empty_cells


class Board(object):

    def __init__(self, height, weight):
        self.board = np.empty((height, weight, 3), dtype=np.uint8)
        self.board[:, :, :] = BoardColor.SPACE_COLOR

    def fill_cell(self, vertex, cell_size, color):
        x, y = vertex
        self.board[x:x+cell_size, y:y+cell_size, :] = color