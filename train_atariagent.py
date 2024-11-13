from atari_agent import AtariAgent

# SET THESE TO CHOOSE WHICH GAME, AND FOR HOW LONG TO TRAIN
TOT_OBSERVATIONS = 20

game = "ALE/Breakout-v5"
number_a = 4 # Check game documentation
where_to_save = "breakout_saved.keras"

# Train
breakout_agent = AtariAgent(game, number_a, where_to_save)
breakout_agent.train(TOT_OBSERVATIONS)