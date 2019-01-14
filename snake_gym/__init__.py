from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_gym.envs:SnakeEnv',
)
register(
    id='MultiSnake-v0',
    entry_point='snake_gym.envs:MultiSnakeEnv',
)