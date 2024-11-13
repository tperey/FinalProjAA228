# use to test out

import gymnasium as gym
import ale_py
import numpy as np

game_name = "ALE/Breakout-v5" # Change to change game

# Print all available environments
print(gym.envs.registry.keys())
gym.register_envs(ale_py)

# Create the game environment
env = gym.make(game_name, render_mode = 'human', full_action_space=False)
cum_reward = 0

# Reset the environment to its initial state
obs = env.reset()

for j in range(1000):
    env.render()  # Render the environment
    action = env.action_space.sample()  # Take a random action

    obs, reward, terminated, truncated, info = env.step(action)  # Take the action and get the new state

    cum_reward += reward

    if terminated or truncated:
        print("Game over!")
        #obs = env.reset()  # Reset if the game ends
        break

env.close()  # Close the environment when done
