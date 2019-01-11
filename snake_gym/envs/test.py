from snake import SnakeEnv
import matplotlib.pyplot as plt 


env = SnakeEnv()
env.seed(0)
s = env.reset()
# env.render()
while(True):
    # plt.imshow(s)
    # plt.show()
    # for i in range(2):
    #     env.step(1)
    #     env.render()
    # for i in range(5):
    #     env.step(2)
    #     env.render()
    # for i in range(10):
    #     env.step(0)
    #     env.render()
    # for i in range(12):
    #     env.step(3)
    #     env.render()
    # for i in range(8):
    #     env.step(1)
    #     env.render()
    # for i in range(15):
    #     env.step(2)
    #     env.render()
    # for i in range(9):
    #     env.step(1)
    #     env.render()
    
    s, rew, done, info = env.step(env.action_space.sample())

    env.render()
    
    if done:
        env.reset()