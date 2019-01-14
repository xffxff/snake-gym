# Snake game as a Gym environment
## Installation
```
git clone https://github.com/XFFXFF/snake-gym  
cd snake-gym  
pip install -e .
```

## How to use it
### Running the classic snake game with random policy
```python
import gym
import snake_gym

env = gym.make('Snake-v0')
obs = env.reset()
while True:
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    env.render()
    if done:
        env.reset()
```

### Running multiple snakes with random policy (now there is only two snakes).
```python
import gym
import snake_gym

env = gym.make('MultiSnake-v0')
obs = env.reset()
while True:
    act = env.action_space.sample() 
    obs, rew, done, info = env.step(act)
    env.render()
```
