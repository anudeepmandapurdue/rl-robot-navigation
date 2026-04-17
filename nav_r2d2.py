from stable_baselines3 import PPO
from env.navigation_env import NavigationEnv

env = NavigationEnv(render=False)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[128, 128]),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=500_000)
model.save("nav_model")