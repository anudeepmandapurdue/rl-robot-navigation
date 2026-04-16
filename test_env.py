from env.navigation_env import NavigationEnv
import numpy as np

environment = NavigationEnv(render=True)
obs = environment.reset()

for i in range(200):
    action = np.array([5.0,0.0])
    obs, reward, done, info = environment.step(action)

    print(reward)

    if done:
        print("Episode finished")
        obs = environment.reset()