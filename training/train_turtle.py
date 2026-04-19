from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.navigation_turtle_env import NavigationEnv


def make_env():
    return NavigationEnv(render=False)


env = make_vec_env(make_env, n_envs=4)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./ppo_nav_logs/"
)

model.learn(total_timesteps=700_000)

model.save("ppo_turtlebot_nav_21")

print("Training complete.")