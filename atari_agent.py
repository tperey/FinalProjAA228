import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
import cv2
import numpy as np
import os
import shutil
from datetime import datetime
import csv
from memory_profiler import memory_usage
import pickle
import gc
import psutil
import tensorflow as tf
from tensorflow.keras import backend as K

from replay_queue import ReplayQueue
from double_dQ import doubleDeepQ

# EMPIRICALLY, with matrix improvements, code runs just as fast on Colab as your computer, basically
# But, your computer ok with larger replay buffers, so lets do that

# Constants and hyperparameters
# REPLAY_SIZE = 20000 # From exs. Paper used 1M.
REPLAY_SIZE = 5000 # Empirically, Trevor's computer can't handle larger than this
#REPLAY_SIZE = 1200 # Empirically, free Google cloud can't handle more than this

MINIBATCH_SIZE = 32 # Number of samples to train on with each iteration. Paper used 32
#MINIBATCH_SIZE = 64 # But, GPU can process more batches at once
#MINIBATCH_SIZE = 128 # Slows down a lot

NUM_FRAMES = 4 # Number of frames stacked into single state input for training and experience replay. 4 was used in DeepMind paper

INITIAL_EPSILON = 1.0 # Starting probability for e-greedy exploration
FINAL_EPSILON = 0.1 # Final probability for exploration. TUNED from DeepMind
#EPSILON_DECAY = 300000 # Old decay rate that gave semi-decent policy after 4k iterations
EPSILON_DECAY_FACTOR = 0.5 # Percentage of tot_frames at which to hit final_epsilon

TAD_EPSILON = 0.1 # For use in simulation, to prevent getting stuck

# MIN_OBSERVATION = 5000 # Number of required states before starting to train
# TOT_OBSERVATION = 1000000 # Number of training states.
# SAVE_OBSERVATION = 10000 # Number of observations after which to save model (for intermediate progress saving)
# Rather than use these...
# -- Tot frames is passed in. Don't need constant for that
# -- Calculate min_observatin and save_observation
MIN_O_DIV = 20 # Divides tot_frames to determine after which observation to start minibatch
SAVE_O_DIV = 20 # Divides tot_frames to determine after which observation to save intermittently.
# Save twice as often

# SMALL_PRINT = 1
# LARGE_PRINT = 10
SMALL_PRINT = 10
LARGE_PRINT = 1000

""" General class for agent that plays an Atari game. """
# Should work with any game
class AtariAgent(object):

    #INIT: specify game and action space size

    def __init__(self, game_name, num_actions, path_to_save_model, how_to_render = 'human'):

        self.save_path = path_to_save_model

        # Make a path for saving iteration, q-value
        save_base, _ = self.save_path.rsplit(".", 1) # Get the front of the save file name
        q_path = save_base + "_qlog.csv" # Add new ending
        self.q_log_path = q_path

        self.env = gym.make(game_name, render_mode = how_to_render, full_action_space=False) # Create the game gym env. Variable of the class. Only relevant actions
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
    
    # LOAD_DEEPQ: bring in existing deepQ model and buffer for continuing training
    def load_deepq(self, path_model, path_queue = ""):
        self.deep_q.load_model(path_model) # Replace model with another model.
        
        if path_queue: # Load queue (if requested)
            with open(path_queue,"rb") as f:
                self.replay_queue = pickle.load(f)

        # Useful for picking up training where left off, or simulating after training
    
    # CONVERT_PROCESS_BUFFER: convert NUM_FRAMES images in process_buffer into a state
    def convert_process_buffer(self):
        # Conver to greyscale, downsample to (84, 84), and put all buffer images into a list
        gray_buffer = [cv2.resize(cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY), (84, 84)) for observ in self.process_buffer]
        gray_buffer = [gimg[:,:, np.newaxis] for gimg in gray_buffer] # Add axis for stacking. No cropping

        return np.concatenate(gray_buffer, axis = 2) # Stack frames
    
    # TRAIN: train agent in game itself for tot_frames
    def train(self, tot_frames, start_eps = INITIAL_EPSILON):

        # Initialize
        observation_num = 0
        init_state = self.convert_process_buffer() # Whatever state in buffer when train called
        epsilon = start_eps
        alive_frame = 0
        total_reward = 0

        min_observation = tot_frames/MIN_O_DIV # Point at which to start minibatch training
        save_observation = tot_frames/SAVE_O_DIV # Point at which to intermittently save

        epsilon_decay = EPSILON_DECAY_FACTOR*tot_frames # Get decay rate

        while observation_num < tot_frames: # Repeat for total training frames

            if observation_num % LARGE_PRINT == (LARGE_PRINT - 1): # Print every many frames
                print("Executing loop ", observation_num)

            """ DECAY LEARNING RATE"""
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/epsilon_decay

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
                print("At iteration", observation_num, "optimal q = ", predict_q_value)

                # Save q values
                with open(self.q_log_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([observation_num, predict_q_value])

            if done: # Handle game end
                print("Game over! Lasted ", alive_frame, " frames")
                print("Earned total reward of ", total_reward)
                self.env.reset() # Keep playing

                alive_frame = 0
                total_reward = 0
            
            # Append new state
            new_state = self.convert_process_buffer() # Just filled, so convert
            self.replay_queue.add(curr_state, predict_a, reward, done, new_state) # Add to replay
            total_reward += reward

            """ TRAIN - minibatch """
            if self.replay_queue.size() > min_observation: # Wait until replay reasonably full

                s_b, a_b, r_b, d_b, s_new_b = self.replay_queue.sample(MINIBATCH_SIZE) # Sample minibatch
                self.deep_q.train(s_b, a_b, r_b, d_b, s_new_b, observation_num) # Train on minibatch
                self.deep_q.target_train() # INTIIALLY FORGOT THIS!!!

                """ Periodically SAVE & CLEAN """
                if observation_num % save_observation == (save_observation-1):
                    print("...Saving model intermediately...")

                    # Save in DIFF file
                    base_path, keras_extension = self.save_path.rsplit(".", 1)
                    modified_save_path = f"{base_path}_c{observation_num}.{keras_extension}"

                    self.deep_q.save_model(modified_save_path)

                    print("For ref, epsilon = ", epsilon)

                    # Clear and reload
                    K.clear_session()
                    self.deep_q.load_model(modified_save_path) # Load model just saved

                    # ERROR LOGGING
                    print("Mem:", memory_usage(-1, interval=1, timeout=1))
                    print(len(self.replay_queue.queue)) # Doesn't require that you wrote your size function right
                    print("Called garbage collection")
                    gc.collect()

                    memory_info = psutil.virtual_memory()
                    print(f"Total memory: {memory_info.total / (1024 ** 3):.2f} GB")
                    print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")
                    print(f"Used memory: {memory_info.used / (1024 ** 3):.2f} GB")
                    print(f"Memory usage: {memory_info.percent}%")

                    # SAVE BUFFER IN CASE OF CRASH
                    with open("replay_queue.pkl", "wb") as f:
                        pickle.dump(self.replay_queue, f)

            # Update trackers
            alive_frame += 1
            observation_num += 1
        
        # Save after training complete
        self.deep_q.save_model(self.save_path)

    # SIMULATE: run game to Game Over once using current model (whether from training or load, more likely)
    # Only save if reward above save_threshold
    def simulate(self, vid_fold = "", vid_prefix = "", save = False, save_threshold = 0):

        # Assumes model ALREADY LOADED

        # CONSIDER RELOADING MODEL EVERY TIME TO ENSURE ONLY SAVING ONE EPISODE
        #self.env = gym.make(game_name, render_mode = how_to_render, full_action_space=False)

        # Initialize
        done = False
        tot_award = 0

        if save: # Check if saving. If so, save video
            temp_video_folder = "temp_video" # Temporary video folder, so can remove video if not agove threshold

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") # Add timestamp to prefix for easy differentiation
            vid_prefix = vid_prefix + "_" + timestamp + "_"

            os.makedirs(temp_video_folder, exist_ok=True) # Make temp folder
            self.env = RecordVideo(self.env, video_folder = temp_video_folder, name_prefix=vid_prefix, episode_trigger = lambda e: e==0) # Only one episode, so def save it
            # CONSIDER E == 0 TO ONLY SAVE ONE EPISODE
            # No name prefix rn
        
        self.env.reset()
        self.env.render() # Show startup

        # # GET GAME STARTED - run one Action = 1 to make sure game starts running (if model predicts initial 0, it won't)
        # self.env.render() #visualize
        # observation, reward, done, _, _ = self.env.step(1)
        # print("STARTUP")
        # self.process_buffer.append(observation) # Make sure process buffer reflects initialization
        # self.process_buffer = self.process_buffer[1:]

        # Run the game
        while not done:

            # Choose action using model
            state = self.convert_process_buffer() # Get current state
            predict_action = self.deep_q.predict_movement(state, TAD_EPSILON)[0] # A little exploratio to prevent getting stuck, and only keep action (not q)
            
            # print(predict_action) # For debugging only
            # Render and get result
            self.env.render() #visualize
            observation, reward, done, _, _ = self.env.step(predict_action)
            tot_award += reward

            # Update process buffer
            self.process_buffer.append(observation)
            self.process_buffer = self.process_buffer[1:] # Remove oldest image in process buffer
        
        print("Game over! Final score = ", tot_award) # Give final score

        if save:
            self.env.close() # save in temp folder
            if tot_award >= save_threshold: # If satisfies threshold
                if vid_fold: # If destination folder specified
                    os.makedirs(vid_fold, exist_ok = True) # Make it (if not already there)

                    for filename in os.listdir(temp_video_folder): # Go through all files in temp folder
                        shutil.move(os.path.join(temp_video_folder, filename), vid_fold) # Move to permanent folder
                
                print("!!!")
                print("!!!")
                print("!!!")
                print(f"Video saved. Reward was {tot_award}, which satisfies thresh = {save_threshold}")
                print("!!!")
                print("!!!")
                print("!!!")
            else: # Otherwise
                print(f"Video NOT saved. Reward was only {tot_award}, which is BELOW thresh = {save_threshold}")
            
            shutil.rmtree(temp_video_folder) # Remove temp folder
            print("~~~Supposedly just cleared temp folder~~~")
        
        return tot_award # Return final score
    
    # PLAY_AGENT: simulate for multiple episodes. Essentially a wrapper on simulate
    def play_agent(self, episode_num = 1, v_f = "", v_p = "", sB = False, sT = 0, gS = False):

        if gS:
            scores = np.zeros(episode_num) # Initialize vector for storing scores

        # Simply call simulate for specified episode num
        for i in range(episode_num):
            print("...Running episode ", i+1)
            last_score = self.simulate(vid_fold = v_f, vid_prefix=v_p, save=sB, save_threshold=sT)

            if gS:
                scores[i] = last_score
        
        if gS:
            print("Average scores: ", np.mean(scores))

    # PLAY_AGENT: simulate for multiple episodes, and report average score
    def get_mean_score(self, episode_num = 1, score_path = ""):

        scores = np.zeros(episode_num) # Initialize vector for storing scores

        # Simply call simulate for specified episode num
        for i in range(episode_num):
            ind_score = self.simulate() # Don't need any parameters, b/c not saving
            
            scores[i] = ind_score

            if score_path: # Save to file, if specified
                with open(score_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([i, ind_score])
                    print(" Saved score ", i+1, " = ", ind_score)
        
        # Get and report averge score
        ave_score = np.mean(scores) 
        print("AVERAGE score = ", ave_score)
        




    # May need MEAN