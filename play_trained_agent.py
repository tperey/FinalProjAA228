import gymnasium as gym
import ale_py
from atari_agent import AtariAgent

import os
from datetime import datetime

""" SET THESE TO CHOOSE WHICH GAME, AND WHICH MODEL """
game = "ALE/Breakout-v5"
number_a = 4 # Check game documentation
model_path = "breakout_saved.keras" # Use for save_path and to load

video_fold = "good_runs_111324"
video_pref = "grun"

saveBool = True

save_thresh = 0
num_trials = 1

""" Simulate 'n' save """
breakout_agent = AtariAgent(game, number_a, model_path, 'rgb_array') # Instantiate
breakout_agent.load_deepq(model_path)
breakout_agent.play_agent(episode_num=num_trials, v_f=video_fold, v_p=video_pref, sB=saveBool, sT=save_thresh)

#def play_agent(self, episode_num = 1, v_f = "", v_p = "", sB = False, sT = 0):
#simulate(self, vid_fold = "", vid_prefix = "", save = False, save_threshold = 0):