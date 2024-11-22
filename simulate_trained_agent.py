import gymnasium as gym
import ale_py
from atari_agent import AtariAgent

import os
from datetime import datetime

""" SET THESE TO CHOOSE WHICH GAME, AND WHICH MODEL """
game = "ALE/UpNDown-v5"
number_a = 6 # Check game documentation
#model_path = "./adv_models/breakout_adv_c999.keras" # Use for save_path and to load
#model_path = "./adv_models/old_breakout_adv_c999.keras"
#model_path = "breakout_saved.keras"
#model_path = "./adv_models/breakout_adv_c7499.keras"
#model_path = "./adv_models/breakout_adv_c6499.keras"
#model_path = "./adv_models/breakout_adv_c3499.keras"
#model_path = "./UNDv3_models/UNDv3_c9999.keras"
model_path = "./UNDv7_models/UNDv7_rr_c21249.keras"


video_fold = "test_out_saving"
video_pref = "6k_failed_testrun"

saveBool = False

save_thresh = 0

""" Simulate 'n' save """
breakout_agent = AtariAgent(game, number_a, model_path, 'human') # Instantiate
breakout_agent.load_deepq(model_path)
breakout_agent.simulate(vid_fold=video_fold, vid_prefix=video_pref, save=saveBool, save_threshold=save_thresh)

#simulate(self, vid_fold = "", vid_prefix = "", save = False, save_threshold = 0):