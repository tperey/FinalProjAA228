import gymnasium as gym
import ale_py
from atari_agent import AtariAgent

""" SET THESE TO CHOOSE WHICH GAME, AND FOR HOW LONG TO TRAIN """
# TOT_OBSERVATIONS = 20
TOT_OBSERVATIONS = 10000
RESTART_EPSILON = 0.1 # Start w/o exploring, since should have stopped but it got killed

game = "ALE/UpNDown-v5"
number_a = 6 # Check game documentation
where_to_save = "./UNDv2_models/UNDv2_7k.keras"

model_to_load = "./UNDv2_models/UNDv2_c6999.keras"
queue_to_load = "replay_queue.pkl"

""" Train """
breakout_agent = AtariAgent(game, number_a, where_to_save, None)
breakout_agent.load_deepq(model_to_load, queue_to_load)
breakout_agent.train(TOT_OBSERVATIONS, RESTART_EPSILON)