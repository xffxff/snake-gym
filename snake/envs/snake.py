
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
    FOOD = 2

class SnakeReward(object):
    ALIVE = 0.
    FOOD = 1.
    DEAD = -1.
    WON = 100.

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
        elif next_head_state == SnakeCellState.FOOD:
            if len(self.empty_cells) > 0:
                self.food = self.empty_cells[self.np_random.choice(len(self.empty_cells))]
                self.empty_cells.remove(self.food)
                reward = SnakeReward.FOOD
            else:
                reward = SnakeReward.WON
        else:
            self.snake.pop()
            reward = 0.
        return self.viewer.get_array(), reward, done, {}

    def next_head(self, action):
        x, y = self.snake_head
        if action == SnakeAction.LEFT:
            return (x - 1, y)
        if action == SnakeAction.RIGHT:
            return (x + 1, y)
        if action == SnakeAction.UP:
            return (x, y + 1)
        return (x, y - 1)

    def cell_state(self, cell):
        if cell in self.empty_cells:
            return SnakeCellState.EMPTY
        if cell == self.food:
            return SnakeCellState.FOOD
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
        screen_width = 400
        screen_height = 400
        cell_size = screen_width / self.width

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        for x, y in self.snake:
            l, r, t, b = x * cell_size, (x + 1) * cell_size, y * cell_size, (y + 1) * cell_size
            snake = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            snake.set_color(0, 0, 0)
            self.viewer.add_onetime(snake)

        if self.food:
            x, y = self.food
            l, r, t, b = x * cell_size, (x + 1) * cell_size, y * cell_size, (y + 1) * cell_size
            food = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            food.set_color(1, 0, 0)
            self.viewer.add_onetime(food)
        return self.viewer.render(return_rgb_array = mode == 'rgb_array')