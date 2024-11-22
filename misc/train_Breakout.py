import gymnasium as gym
import ale_py
from atari_agent import AtariAgent

""" SET THESE TO CHOOSE WHICH GAME, AND FOR HOW LONG TO TRAIN """
# TOT_OBSERVATIONS = 20
TOT_OBSERVATIONS = 10000

game = "ALE/Breakout-v5"
number_a = 4 # Check game documentation
where_to_save = "./Bv3_models/bv3.keras"

""" Train """
breakout_agent = AtariAgent(game, number_a, where_to_save, None)
breakout_agent.train(TOT_OBSERVATIONS)