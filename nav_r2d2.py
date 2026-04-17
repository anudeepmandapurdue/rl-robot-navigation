
from stable_baselines3 import PPO
from env.navigation_env import NavigationEnv


environment = NavigationEnv(render = False)
model = PPO("MlpPolicy", environment, verbose=1)

model.learn(total_timesteps=200_000)
model.save("nav_model")