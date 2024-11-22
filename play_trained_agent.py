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
#model_path = "./adv_models/breakout_adv_c6499.keras"
#model_path = "./adv_models/breakout_adv_c3999.keras"
# model_path = "./UNDv3_models/UNDv3_c9999.keras"
model_path = "./UNDv7_models/UNDv7_rr_c21249.keras"

video_fold = "UNDv7_goodruns"
video_pref = "r2k100_UNDv7_"

saveBool = True
getScores = True

save_thresh = 6000
num_trials = 100

""" Simulate 'n' save """
breakout_agent = AtariAgent(game, number_a, model_path, 'rgb_array') # Instantiate
breakout_agent.load_deepq(model_path)
breakout_agent.play_agent(episode_num=num_trials, v_f=video_fold, v_p=video_pref, sB=saveBool, sT=save_thresh, gS=getScores)

#def play_agent(self, episode_num = 1, v_f = "", v_p = "", sB = False, sT = 0):
#simulate(self, vid_fold = "", vid_prefix = "", save = False, save_threshold = 0):