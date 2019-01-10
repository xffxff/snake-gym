from snake import SnakeEnv


env = SnakeEnv()
env.seed(0)
env.reset()
# env.render()
while(True):
    s, rew, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        env.reset()