
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
# model_path = "./UNDv3_models/UNDv3_c9999.keras"
model_path = "./UNDv7_models/UNDv7_rr_c21249.keras"

scorePath = "scores_UNDv7_100kv1.csv"

num_trials = 100

""" Simulate 'n' score """
breakout_agent = AtariAgent(game, number_a, model_path, 'rgb_array') # Instantiate
breakout_agent.load_deepq(model_path)
breakout_agent.get_mean_score(episode_num=num_trials, score_path=scorePath)