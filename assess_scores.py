
import gymnasium as gym
import ale_py
from atari_agent import AtariAgent

import os
from datetime import datetime


""" SET THESE TO CHOOSE WHICH GAME, AND WHICH MODEL """
game = "ALE/UpNDown-v5"
number_a = 6 # Check game documentation
#model_path = "breakout_saved.keras" # Use for save_path and to load
# model_path = "./adv_models/old_breakout_adv_c999.keras"
model_path = "./UNDv1_models/UNDv1_c4499.keras"

scorePath = "scores_UNDv1_c4499.csv"

num_trials = 20

""" Simulate 'n' score """
breakout_agent = AtariAgent(game, number_a, model_path, 'rgb_array') # Instantiate
breakout_agent.load_deepq(model_path)
breakout_agent.get_mean_score(episode_num=num_trials, score_path=scorePath)