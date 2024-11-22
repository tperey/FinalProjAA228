import gymnasium as gym
import ale_py
from atari_agent import AtariAgent

""" SET THESE TO CHOOSE WHICH GAME, AND FOR HOW LONG TO TRAIN """
# TOT_OBSERVATIONS = 20
TOT_OBSERVATIONS = 25000
RESTART_EPSILON = 0.5 # Restart w/ exploring

game = "ALE/UpNDown-v5"
number_a = 6 # Check game documentation
where_to_save = "./UNDv7_models/UNDv7_rr.keras"

model_to_load = "./UNDv6_models/UNDv6_r_c19999.keras"
queue_to_load = "replay_queue.pkl"

""" Train """
breakout_agent = AtariAgent(game, number_a, where_to_save, None)
breakout_agent.load_deepq(model_to_load, queue_to_load)
breakout_agent.train(TOT_OBSERVATIONS, RESTART_EPSILON)