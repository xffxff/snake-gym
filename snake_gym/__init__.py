from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_gym.envs:SnakeEnv',
)
# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv',
# )