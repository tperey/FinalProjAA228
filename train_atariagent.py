import gymnasium as gym
import ale_py
from atari_agent import AtariAgent

""" SET THESE TO CHOOSE WHICH GAME, AND FOR HOW LONG TO TRAIN """
# TOT_OBSERVATIONS = 20
TOT_OBSERVATIONS = 10000

game = "ALE/Breakout-v5"
number_a = 4 # Check game documentation
where_to_save = "./adv_models/breakout_adv.keras"

""" Comment in to confirm getting correct game """
# # Create the game environment
# env = gym.make(game, render_mode = 'human', full_action_space=False)
# cum_reward = 0

# # Reset the environment to its initial state
# obs = env.reset()

# for j in range(1000):
#     env.render()  # Render the environment
#     action = env.action_space.sample()  # Take a random action

#     obs, reward, terminated, truncated, info = env.step(action)  # Take the action and get the new state

#     cum_reward += reward

#     if terminated or truncated:
#         print("Game over!")
#         #obs = env.reset()  # Reset if the game ends
#         break

# env.close()  # Close the environment when done


""" Train"""
breakout_agent = AtariAgent(game, number_a, where_to_save, None)
breakout_agent.train(TOT_OBSERVATIONS)