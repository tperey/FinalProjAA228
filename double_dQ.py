"""dDeepQ: class for implementing double deep Q learning"""

import gymnasium
import numpy as np
import random
import tensorflow as tf
import cv2

#import GPUtil

from replay_queue import ReplayQueue

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Constants and hyperparameters
NUM_FRAMES = 4 # Number of frames stacked into single state input for training and experience replay. 4 was used in DeepMind paper

LEARN_RATE = 0.00001 # Learning rate. From exs.
DECAY_RATE = 0.99 # Gamma. From exs.

TAU = 0.01 # Weights for soft target CNN update. From exs

# SMALL_PRINT = 1
SMALL_PRINT = 10


class doubleDeepQ(object):

    # INIT: just call construction for simplicity
    def __init__(self, specific_n_a):
        self.num_actions = specific_n_a # Input num_actions (rather than constant) to make general (for any game)
        self.construct_q_network()
        #tf.get_logger().setLevel('ERROR') # Don't print bars every time
    
    # CONSTRUCT_Q_NETWORK: construct both standard and TARGET CNNs for use in double Q learning
    def construct_q_network(self):

        # Uses structure from DeepMind 2015 paper
        # Massively Parallel Methods for Deep Reinforcement Learning

        """ Online CNN """
        # This is the one being continously learned.
        self.model = Sequential()

        # Input layer
        self.model.add(Input(shape=(84, 84, NUM_FRAMES)))

        # Convulational layers
        self.model.add(Conv2D(32, (8,8), strides = (4,4), activation = 'relu')) # First layer -> 32 filters, 8 x 8, stride size of 4 in both directions, relu activation, 
        self.model.add(Conv2D(64, (4,4), strides = (2,2), activation = 'relu')) # Second layer -> 64 filters, 4 x 4, strides of 2, relu
        self.model.add(Conv2D(64, (3,3), strides = (1,1), activation = 'relu')) # Third layer -> 64 fitlers, 3 x 3, stride 1, relu

        # Fully connected layers
        self.model.add(Flatten())
        self.model.add(Dense(512, activation = 'relu')) # Initial fully-connected to process features from CNN
        self.model.add(Dense(self.num_actions, activation = 'linear')) # Final layer. Output Q values for each action given state

        # Set training options - use MSE loss, ADAM optimizer with LEARN_RATE
        self.model.compile(loss = 'mse', optimizer = Adam(learning_rate = LEARN_RATE))



        """ Target CNN """
        # For generating targets. Per double DQN algorithm
        # Same structure as online
        self.target_model = Sequential()

        # Input layer
        self.target_model.add(Input(shape=(84, 84, NUM_FRAMES)))

        # Convulational layers
        self.target_model.add(Conv2D(32, (8,8), strides = (4,4), activation = 'relu')) # First layer -> 32 filters, 8 x 8, stride size of 4 in both directions, relu activation, 
        self.target_model.add(Conv2D(64, (4,4), strides = (2,2), activation = 'relu')) # Second layer -> 64 filters, 4 x 4, strides of 2, relu
        self.target_model.add(Conv2D(64, (3,3), strides = (1,1), activation = 'relu')) # Third layer -> 64 fitlers, 3 x 3, stride 1, relu

        # Fully connected layers
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512, activation = 'relu')) # Initial fully-connected to process features from CNN
        self.target_model.add(Dense(self.num_actions, activation = 'linear')) # Final layer. Output Q values for each action given state

        # Set training options - use MSE loss, ADAM optimizer with LEARN_RATE
        self.target_model.compile(loss = 'mse', optimizer = Adam(learning_rate = LEARN_RATE))

        self.target_model.set_weights(self.model.get_weights()) # Ensure both models have same initial weights for proper training/matching

        print("Successfully constructed online and target CNNs") # Check

    # PREDICT_MOVEMENT: determine next mvmnt given data (state, aka stack of images)
    # model, and epsilon-greedy exploration policy
    def predict_movement(self, data, epsilon):
        
        # Get Q values associated with state, and extract optimal action
        q_actions = self.model.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size = 1, verbose = 0) # 'predict' expects BATCHES of samples (states), so reshape
        opt_action = np.argmax(q_actions) # Extract optimal action

        # With probability epsilon, choose random action
        rand_chance = np.random.random() # Randomly choose # from 0 to 1
        if rand_chance < epsilon: # Chance of this occuring is epsilon
            opt_action = np.random.randint(0, self.num_actions)
        
        # Return opt_action and its q value
        return opt_action, q_actions[0, opt_action]

    # TRAIN: given a batch of (s, a, r, d, s_new), retrain the model
    def train(self, s_batch, a_batch, r_batch, d_batch, s_new_batch, obsv_num):
        
        batch_size = s_batch.shape[0] # Number of samples is first dim size of any input
        #targets = np.zeros((batch_size, self.num_actions)) # Initialize targets array

        # Compute targets according to DDQN algorithm (DeepMind Double Q 2015)
        # In general, targets are Q values updated per Q-learning algorithm
        # Computed using models, and DDQN specifies which model to use and when

        # BATCH PROCESS TO MORE EFFECTIVELY USE GPUs
        #print("Data type of s_batch:", s_batch.dtype)
        #print("Shape of s_batch:", s_batch.shape)
        #s_batch = tf.expand_dims(s_batch, axis=1) # Reshape
        #s_new_batch = tf.expand_dims(s_new_batch, axis=1) #Reshape
        #print("~~~DID RESHAPE WITH TF~~~")
        #print("Data type of s_batch:", s_batch.dtype)
        #print("Shape of s_batch:", s_batch.shape)

        q_curr_online = self.model.predict(s_batch, batch_size = 1, verbose = 0) # Initialize to current model predictions, for unseen actions
        # Ensures Q values for unseen (s,a) pairs unchanged as desired

        # Per DDQN, use online CNN to determine best next action (a giving max Q for s_new)
        q_next_online = self.model.predict(s_new_batch, batch_size = 1, verbose = 0)
        best_next_actions = np.argmax(q_next_online, axis = 1) # Get optimal actions across samples

        # Per DDQN, evaluate Q of next state with target CNN
        fut_q = self.target_model.predict(s_new_batch, batch_size = 1, verbose = 0)

        targets = q_curr_online.copy()
        for i in range(batch_size): # For each state in batch, update targets
            # Build up Q for ith observed (s,a) pair
            targets[i, a_batch[i]] = r_batch[i] # Add in reward
            if d_batch[i] == False: # Confirm NOT done (i.e. game over)
                targets[i, a_batch[i]] += DECAY_RATE * fut_q[i, best_next_actions[i]] # Use Q assoc w/ best action for sample i

        # Retrain
        loss = self.model.train_on_batch(s_batch, targets)

        # Print loss every 10 iterations
        if obsv_num % SMALL_PRINT == 0:
            print("At iteration ", obsv_num, "Loss = ", loss)
            print("")
            #GPUtil.showUtilization() # Confirm using GPU memory
    
    # SAVE_MODEL: save DDQN model to file
    def save_model(self, path):
        self.model.save(path)
        print("SAVED model.")
    
    # LOAD_MODEL: load existing DDQN model from file
    def load_model(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print("Successfully loaded network")
    
    # TARGET_TRAIN: train target CNN with soft updates
    def target_train(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(model_weights)): # for each weight
            # Soft update - heavily weight current target CNN weights, but add in lightly-weighted online CNN weights
            target_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_weights[i]

        self.target_model.set_weights(target_weights) # Update target model

""" TEST SCRIPT """
if __name__ == "__main__":
    print("") # Skip a line

    spec_num_a = 6
    test = doubleDeepQ(spec_num_a)