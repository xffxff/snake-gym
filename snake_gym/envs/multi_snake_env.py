
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


FOOD_COLOR = np.array([0, 255, 0], dtype=np.uint8)
SPACE_COLOR = np.array([255,255,255], dtype=np.uint8)
OPPONENT_COLOR = np.array([0, 0, 0], dtype=np.uint8)

SNAKE_COLOR = [np.array([255, 0, 0], dtype=np.uint8), \
               np.array([0, 0, 255], dtype=np.uint8), \
               np.array([255, 255, 0], dtype=np.uint8), \
               np.array([0, 255, 255], dtype=np.uint8), \
               np.array([255, 0, 255], dtype=np.uint8)]



unique_list = lambda x: list(set(x))


class MultiSnakeEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):
        self.width = 15
        self.height = 15
        self.cell_size = 10
        self.board_width = self.width * self.cell_size
        self.board_height = self.height * self.cell_size
        default_n_snakes = 2
        default_n_foods = 3

        self.action_space = spaces.Tuple([spaces.Discrete(4) for i in range(default_n_snakes)])
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=255, 
                            shape=(self.board_width, self.board_height, 3), dtype=np.uint8) 
                            for i in range(default_n_snakes)])

        self.n_snakes = default_n_snakes
        self.snake_alive_num = default_n_snakes
        self.n_foods = default_n_foods
        
        self.foods = []
        self.viewer = None
        self.np_random = np.random
        self.game_over = False
    
    def set_foods(self, n):
        self.n_foods = n
    
    def set_snakes(self, n):
        self.n_snakes = n

    def reset(self):
        self.snake_alive_num = self.n_snakes
        self.game_over = False
        self.snakes = [Snake(i) for i in range(self.n_snakes)]
        empty_cells = self.get_empty_cells()
        for i in range(self.n_snakes):
            empty_cells = self.snakes[i].reset(empty_cells, self.np_random)
        self.foods = [empty_cells[i] for i in self.np_random.choice(len(empty_cells), self.n_foods)]
        return self.get_observations()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        for i in range(self.n_snakes):
            self.snakes[i].step(action[i])
        
        for snake in self.snakes:
            if not snake.done:
                if self.is_collided_wall(snake.head):
                    snake.reward -= 1
                    snake.done = True
                    self.foods.extend(list(snake.body)[1:])
                
                elif self.bite_others_or_itself(snake):
                    snake.reward -= 1
                    snake.done = True
                    self.foods.extend(list(snake.body)[1:])
                
                elif snake.head in self.foods:
                    snake.reward += 1.
                    snake.grow()
                    self.foods.remove(snake.head)
                    self.foods = unique_list(self.foods)
                    empty_cells = self.get_empty_cells()
                    if len(self.foods) < 10:
                        food = empty_cells[self.np_random.choice(len(empty_cells))]
                        self.foods.append(food) 
        
        rewards = []
        dones = [] 
        steps = []
        
        snake_alive_num = self.n_snakes
        for snake in self.snakes:   
            rewards.append(snake.reward)
            dones.append(snake.done)
            steps.append(snake.n_steps)
            snake.reward = 0.
            if snake.done:
                snake_alive_num -= 1
                snake.die()

        if snake_alive_num == 0:
            self.game_over = True
        return self.get_observations(), rewards, dones, {'game_over': self.game_over, 'steps': steps}
    
    def bite_others_or_itself(self, this_snake):
        snakes = self.snakes.copy()
        other_snakes = snakes.remove(this_snake)
        for snake in snakes:
            if this_snake.head == snake.prev_head and self.is_opposite_movement(this_snake, snake):
                return True
        all_body_cells = []
        for snake in snakes:
            all_body_cells.extend(list(snake.body))
        all_body_cells.extend(list(this_snake.body)[1:])
        return this_snake.head in all_body_cells

    def is_opposite_movement(self, snake1, snake2):
        return (snake1.prev_action, snake2.prev_action) in [(SnakeAction.LEFT, SnakeAction.RIGHT),\
                                                            (SnakeAction.RIGHT, SnakeAction.LEFT),\
                                                            (SnakeAction.UP, SnakeAction.DOWN),\
                                                            (SnakeAction.DOWN, SnakeAction.UP)]

    def get_image(self):
        board = Board(self.board_height, self.board_width)
        for snake in self.snakes:
            for x, y in snake.body:
                board.fill_cell((x*self.cell_size, y*self.cell_size), self.cell_size, snake.color)
        
        for food in self.foods:
            x, y = food
            board.fill_cell((x*self.cell_size, y*self.cell_size), self.cell_size, FOOD_COLOR)
        return board.board

    def get_observations(self):
        observations = []
        for snake in self.snakes:
            if snake.done == True:
                observations.append(np.zeros((self.board_width, self.board_height, 3), dtype=np.uint8))
            else:
                board = Board(self.board_height, self.board_width)
                other_snakes = self.snakes.copy()
                other_snakes.remove(snake)
                for x, y in snake.body:
                    board.fill_cell((x*self.cell_size, y*self.cell_size), self.cell_size, SNAKE_COLOR[0])
                for other_snake in other_snakes:
                    for x, y in other_snake.body:
                        board.fill_cell((x*self.cell_size, y*self.cell_size), self.cell_size, OPPONENT_COLOR)
                for food in self.foods:
                    x, y = food
                    board.fill_cell((x*self.cell_size, y*self.cell_size), self.cell_size, FOOD_COLOR)
                observations.append(board.board)
        return observations

    def get_empty_cells(self):
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        for snake in self.snakes:
            for cell in snake.body:
                if cell in empty_cells:
                    empty_cells.remove(cell)
        for food in self.foods:
            if food in empty_cells:
                empty_cells.remove(food)
        return empty_cells

    def is_collided_wall(self, head):
        x, y = head
        if x < 0 or x > (self.width - 1) or y < 0 or y > (self.height - 1):
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

    def __init__(self, i):
        self.body = deque()
        self.color = SNAKE_COLOR[i]
        self.prev_action = None
        self.prev_head = None
        self.tail = None
        self.reward = 0.
        self.done = False
        self.n_steps = 0
        
    def step(self, action):
        if not self.done:
            if not self.is_valid_action(action):
                action = self.prev_action
            self.prev_action = action
            self.prev_head = self.head
            x, y = self.head
            if action == SnakeAction.LEFT:
                self.body.appendleft((x, y - 1))
            if action == SnakeAction.RIGHT:
                self.body.appendleft((x, y + 1))
            if action == SnakeAction.UP:
                self.body.appendleft((x - 1, y))
            if action == SnakeAction.DOWN:
                self.body.appendleft((x + 1, y))
            self.tail = self.body.pop()
            self.n_steps += 1
        return action
    
    def grow(self):
        self.body.append(self.tail)
    
    def die(self):
        self.body.clear()

    @property
    def head(self):
        return self.body[0]

    def is_valid_action(self, action):
        if len(self.body) == 1:
            return True
        
        horizontal_actions = [SnakeAction.LEFT, SnakeAction.RIGHT]
        vertical_actions = [SnakeAction.UP, SnakeAction.DOWN]

        if self.prev_action in horizontal_actions:
            return action in vertical_actions
        return action in horizontal_actions
    
    def reset(self, empty_cells, np_random):
        self.reward = 0.
        self.done = False
        self.body.clear()
        start_head = empty_cells[np_random.choice(len(empty_cells))]
        self.body.appendleft(start_head)
        empty_cells.remove(start_head)
        return empty_cells


class Board(object):

    def __init__(self, height, weight):
        self.board = np.empty((height, weight, 3), dtype=np.uint8)
        self.board[:, :, :] = SPACE_COLOR

    def fill_cell(self, vertex, cell_size, color):
        x, y = vertex
        self.board[x:x+cell_size, y:y+cell_size, :] = color

