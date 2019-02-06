from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_gym.envs:SnakeEnv',
)

register(
    id='Snake-rgb-v0',
    entry_point='snake_gym.envs:SnakeEnv',
    kwargs={'observation_mode': 'rgb'}
)

register(
    id='MultiSnake-v0',
    entry_point='snake_gym.envs:MultiSnakeEnv',
)
