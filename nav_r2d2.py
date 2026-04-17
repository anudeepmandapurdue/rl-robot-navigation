
from stable_baselines3 import PPO
from env.navigation_env import NavigationEnv


environment = NavigationEnv(render = False)
policy_kwargs = dict(net_arch=[64, 64])

model = PPO(
    "MlpPolicy",
    environment,
    policy_kwargs=policy_kwargs,
    verbose=1
)
model.learn(total_timesteps=200_000)
model.save("nav_model")