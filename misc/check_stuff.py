import gymnasium as gym
import ale_py
from atari_agent import AtariAgent

import os
from datetime import datetime

""" SET THESE TO CHOOSE WHICH GAME, AND WHICH MODEL """
game = "ALE/UpNDown-v5"
number_a = 6 # Check game documentation
model_path = "./UNDv2_models/UNDv2_c6999.keras"
deq_path = "replay_queue.pkl"

""" Check deal with buffer """
breakout_agent = AtariAgent(game, number_a, model_path, 'rgb_array') # Instantiate
breakout_agent.load_deepq(model_path, deq_path)
print(len(breakout_agent.replay_queue.queue))

#def play_agent(self, episode_num = 1, v_f = "", v_p = "", sB = False, sT = 0):
#simulate(self, vid_fold = "", vid_prefix = "", save = False, save_threshold = 0):