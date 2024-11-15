# Run game with random policy, save video and mean score

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
import numpy as np
import csv
from datetime import datetime
import os

#game_name = "ALE/Breakout-v5" # Change to change game

game_name = "ALE/UpNDown-v5" # Change to change game

# Use to run one simulation
def simulate_random(game_env):
    game_env.reset()
    #game_env.render()
    done = False
    cum_reward = 0

    while not done:
        
        action = env.action_space.sample()  # Take a random action
        #game_env.render()  # Render the environment

        obs, reward, done, _, _ = game_env.step(action)  # Take the action and get the new state

        cum_reward += reward
    
    print("Final RANDOM score = ", cum_reward)
    game_env.close()

    return cum_reward


""" MEAN SCORE DETERMINATION """
# Create the game environment
env = gym.make(game_name, render_mode = 'rgb_array', full_action_space=False)

# Get average score
episode_num = 20
scores = np.zeros(episode_num)

for i in range(episode_num):
    scores[i] = simulate_random(env)

    score_path = game_name.split("/")[-1] + "_random_scores.csv"
    with open(score_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i, scores[i]])
        print(" Saved rand score ", i+1, " = ", scores[i])

ave_score = np.mean(scores)
print("Ave RAND score = ", ave_score)

""" SAVE ONE VIDEO """
# Create the game environment
env = gym.make(game_name, render_mode = 'rgb_array', full_action_space=False) # now, render it

done = False
tot_award = 0

vid_folder = "random_video" + game_name.split("/")[-1] # Temporary video folder, so can remove video if not agove threshold
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") # Add timestamp to prefix for easy differentiation
vid_prefix = "random_" + game_name.split("/")[-1] + timestamp + "_"
os.makedirs(vid_folder, exist_ok=True) # Make random folder
env = RecordVideo(env, video_folder = vid_folder, name_prefix=vid_prefix, episode_trigger = lambda e: e==0) # Only one episode, so def save it

simulate_random(env) # Run the game