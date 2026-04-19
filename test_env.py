from stable_baselines3 import PPO
from env.navigation_turtle_env import NavigationEnv

# Create env with rendering
env = NavigationEnv(render=True)

# Load trained model
model = PPO.load("ppo_turtlebot_nav_18")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)

    if done or truncated:
        obs, _ = env.reset()