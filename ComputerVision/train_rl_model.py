import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from robocup_ssl_env import RoboCupSSLEnv

# Create environment
env = RoboCupSSLEnv()

# Check if the environment follows the Gym interface
check_env(venv)

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_robocup_ssl")

# To reload the trained model
# model = PPO.load("ppo_robocup_ssl")

# Evaluate the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
