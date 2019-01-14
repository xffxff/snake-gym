

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


class MultiSnakeEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 50
    }

    def __init__(self):
        self.width = 20
        self.hight = 20

        self.action_space = spaces.Box(low=0, high=3, shape=(2, ), dtype=np.int32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(400, 400, 3), dtype=np.uint8)

        self.snake1 = None
        self.snake2 = None
        self.snakes = []
        self.snake1_prev_act = None
        self.snake2_prev_act = None
        self.food = None
        self.viewer = None
        self.np_random = np.random

    def reset(self):
        self.snake1 = self.snake_rebirth()
        self.snakes.append(self.snake1)
        self.snake2 = self.snake_rebirth()
        self.snakes.append(self.snake2)
        empty_cells = self.get_empty_cells()
        self.food = empty_cells[self.np_random.choice(len(empty_cells))]
        return self.get_image()

    def snake_rebirth(self):
        snake = Snake()
        empty_cells = self.get_empty_cells()
        empty_cells = snake.init(empty_cells, self.np_random)
        return snake

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        snake1_action, snake2_action = action
        if not self.is_valid_action(self.snake1, self.snake1_prev_act, snake1_action):
            snake1_action = self.snake1_prev_act
        if not self.is_valid_action(self.snake2, self.snake2_prev_act, snake2_action):
            snake2_action = self.snake2_prev_act
        self.snake1_prev_act = snake1_action
        self.snake2_prev_act = snake2_action

        snake1_tail = self.snake1.step(snake1_action)
        snake2_tail = self.snake2.step(snake2_action)

        reward1, reward2 = 0., 0.
        done1, done2 = False, False
        #Two snakes collided together
        if self.snake1.head == self.snake2.head:
            reward1 -= len(self.snake1.snake) 
            reward2 -= len(self.snake2.snake)
            done1, done2 = True, True
            self.reset()
        
        #snake1 collided snake2
        if self.snake1.head in self.snake2.body:
            reward1 -= len(self.snake1.snake)
            reward2 += len(self.snake1.snake)
            done1 = True
            done2 = False
            self.snake1 = self.snake_rebirth()
        
        #snake2 collided snake1
        if self.snake2.head in self.snake1.body:
            reward1 += len(self.snake2.snake)
            reward2 -= len(self.snake2.snake)
            done1 = False
            done2 = True
            self.snake2 = self.snake_rebirth()

        if self.snake1.head != self.snake2.head:
            if self.snake1.head == self.food:
                reward1 += 1.
                self.snake1.snake.append(snake1_tail)
                empty_cells = self.get_empty_cells()
                self.food = empty_cells[self.np_random.choice(len(empty_cells))]
            if self.snake2.head == self.food:
                reward2 += 1
                self.snake2.snake.append(snake2_tail)
                empty_cells = self.get_empty_cells()
                self.food = empty_cells[self.np_random.choice(len(empty_cells))]
        
        #two snakes collided wall at the same time
        if self.is_collided_wall(self.snake1.head) and self.is_collided_wall(self.snake2.head):
            reward1 -= len(self.snake1.snake)
            reward2 -= len(self.snake2.snake)
            done1 = True
            done2 = True
            self.snake1 = self.snake_rebirth()
            self.snake2 = self.snake_rebirth()

        #snake1 collided wall
        elif self.is_collided_wall(self.snake1.head):
            reward1 -= len(self.snake1.snake)
            reward2 += len(self.snake1.snake)
            done1 = True
            self.snake1 = self.snake_rebirth()
        
        #snake2 collided wall
        elif self.is_collided_wall(self.snake2.head):
            reward1 += len(self.snake2.snake)
            reward2 -= len(self.snake2.snake)
            done2 = True
            self.snake2 = self.snake_rebirth()

        return self.get_image(), [reward1, reward2], [done1, done2], {}

    def get_image(self):
        board_width = 400
        board_height = 400
        cell_size = int(board_width / self.width)

        board = Board(board_height, board_width)
        for x, y in self.snake1.body:
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.BODY_COLOR)

        for x, y in self.snake2.body:
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.BODY_COLOR)

        x, y = self.snake1.head
        board.fill_cell((x*cell_size, y*cell_size), cell_size, color=BoardColor.HEAD1_COLOR)

        x, y = self.snake2.head
        board.fill_cell((x*cell_size, y*cell_size), cell_size, color=BoardColor.HEAD2_COLOR)
        
        if self.food:
            x, y = self.food
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.FOOD_COLOR)
        return board.board

    def get_empty_cells(self):
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.hight)]
        for snake in self.snakes:
            for cell in snake.snake:
                if cell in empty_cells:
                    empty_cells.remove(cell)
        if self.food in empty_cells:
            empty_cells.remove(self.food)
        return empty_cells

    def is_valid_action(self, snake, prev_action, action):
        if len(snake.snake) == 1:
            return True
        
        horizontal_actions = [SnakeAction.LEFT, SnakeAction.RIGHT]
        vertical_actions = [SnakeAction.UP, SnakeAction.DOWN]

        if prev_action in horizontal_actions:
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