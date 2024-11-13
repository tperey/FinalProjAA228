import gymnasium as gym
import ale_py
import cv2
import numpy as np

from replay_queue import ReplayQueue
from double_dQ import doubleDeepQ

# Constants and hyperparameters
REPLAY_SIZE = 20000 # From exs. Paper used 1M
MINIBATCH_SIZE = 32 # Number of samples to train on with each iteration. Paper used 32

NUM_FRAMES = 4 # Number of frames stacked into single state input for training and experience replay. 4 was used in DeepMind paper

INITIAL_EPSILON = 1.0 # Starting probability for e-greedy exploration
FINAL_EPSILON = 0.01 # Final probability for exploration. TUNED from DeepMind
EPSILON_DECAY = 30000 # Decay rate of exploration probability

# MIN_OBSERVATION = 5000 # Number of required states before starting to train
# TOT_OBSERVATION = 1000000 # Number of training states.
# SAVE_OBSERVATION = 10000 # Number of observations after which to save model (for intermediate progress saving)
# Rather than use these...
# -- Tot frames is passed in. Don't need constant for that
# -- Calculate min_observatin and save_observation
MIN_O_DIV = 20 # Divides tot_frames to determine after which observation to start minibatch
SAVE_O_DIV = 10 # Divides tot_frames to determine after which observation to save intermittently

# SMALL_PRINT = 1
# LARGE_PRINT = 10
SMALL_PRINT = 10
LARGE_PRINT = 1000

""" General class for agent that plays an Atari game. """
# Should work with any game
class AtariAgent(object):

    #INIT: specify game and action space size

    def __init__(self, game_name, num_actions, path_to_save_model):

        self.save_path = path_to_save_model

        self.env = gym.make(game_name, full_action_space=False) # Create the game gym env. Variable of the class. Only relevant actions
        self.env.reset()

        self.replay_queue = ReplayQueue(REPLAY_SIZE) # Create replay queue

        self.deep_q = doubleDeepQ(num_actions) # Init double deep Q network with correct number of action outputs

        # PROCESS BUFFER: for storing last NUM_FRAMES images (observations)
        # Used to generate current state
        self.process_buffer = [] # Init empty list
        # Get initial state
        for i in range(NUM_FRAMES): # Do as for loop, so can easily change NUM_FRAMES
            sc, rc, _, _, _ = self.env.step(0) # Just generate with NOOP
            self.process_buffer.append(sc)
    
    # LOAD_DEEPQ: bring in existing deepQ model
    def load_deepq(self, path):
        self.deep_q.load_model(path) # Replace model with another model.
        # Useful for picking up training where left off, or simulating after training
    
    # CONVERT_PROCESS_BUFFER: convert NUM_FRAMES images in process_buffer into a state
    def convert_process_buffer(self):
        # Conver to greyscale, downsample to (84, 84), and put all buffer images into a list
        gray_buffer = [cv2.resize(cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY), (84, 84)) for observ in self.process_buffer]
        gray_buffer = [gimg[:,:, np.newaxis] for gimg in gray_buffer] # Add axis for stacking. No cropping

        return np.concatenate(gray_buffer, axis = 2) # Stack frames
    
    # TRAIN: train agent in game itself for tot_frames
    def train(self, tot_frames):

        # Initialize
        observation_num = 0
        init_state = self.convert_process_buffer() # Whatever state in buffer when train called
        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        min_observation = tot_frames/MIN_O_DIV # Point at which to start minibatch training
        save_observation = tot_frames/SAVE_O_DIV # Point at which to intermittently save


        while observation_num < tot_frames: # Repeat for total training frames

            if observation_num % LARGE_PRINT == (LARGE_PRINT - 1): # Print every many frames
                print("Executing loop ", observation_num)

            """ DECAY LEARNING RATE"""
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            """ EXECUTE STATE UPDATE """
            curr_state = self.convert_process_buffer() # Get current state (84, 84, NUM_FRAMES)
            self.process_buffer = [] # Clear process_buffer

            predict_a, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon) # Get next move with e-greedy exploration

            reward, done = 0, False # Initialize
            for i in range(NUM_FRAMES): # Execute NUM_FRAMES steps with same action
                temp_obsv, temp_rwd, temp_done, _, _ = self.env.step(predict_a) 
                reward += temp_rwd
                self.process_buffer.append(temp_obsv) # Store new state in process buffer
                done = done | temp_done # True if temp_done = 1 in any state
            
            if observation_num % SMALL_PRINT == 0: # Print q value every so many iterations
                print("At iteration", observation_num, "q = ", predict_q_value)

            if done: # Handle game end
                print("Game over! Lasted ", alive_frame, " frames")
                print("Earned total reward of ", total_reward)
                self.env.reset() # Keep playing

                alive_frame = 0
                total_reward = 0
            
            new_state = self.convert_process_buffer() # Just filled, so convert
            self.replay_queue.add(curr_state, predict_a, reward, done, new_state) # Add to replay
            total_reward += reward


            """ TRAIN - minibatch """
            if self.replay_queue.size() > min_observation: # Wait until replay reasonably full
                s_b, a_b, r_b, d_b, s_new_b = self.replay_queue.sample(MINIBATCH_SIZE) # Sample minibatch
                self.deep_q.train(s_b, a_b, r_b, d_b, s_new_b, observation_num) # Train on minibatch

                # Periodically save model (to avoid losing work)
                if observation_num % save_observation == (save_observation-1):
                    print("...Saving model intermediately...")
                    self.deep_q.save_model(self.save_path)
            
            # Update trackers
            alive_frame += 1
            observation_num += 1
        
        # Save after training complete
        self.deep_q.save_model(self.save_path)

    # will need SIMULATE

    # May need MEAN