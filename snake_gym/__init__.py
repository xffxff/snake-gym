from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_gym.envs:SnakeEnv',
)

register(
    id='MultiSnake-2-v0',
    entry_point='snake_gym.envs:MultiSnakeEnv',
    kwargs={'n_snakes': 2}
)

register(
    id='MultiSnake-3-v0',
    entry_point='snake_gym.envs:MultiSnakeEnv',
    kwargs={'n_snakes': 3}
)

register(
    id='MultiSnake-4-v0',
    entry_point='snake_gym.envs:MultiSnakeEnv',
    kwargs={'n_snakes': 4}
)