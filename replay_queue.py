""" EXPERIENCE REPLAY: implement class for storing set of experienced states for repeated sampling/training (experience replay) """

from collections import deque # Will store experiences in a deque (two-sided queue)
import random
import numpy as np

class ReplayQueue:

    # INIT: create initial queue of requested size
    def __init__(self, queue_size):
        self.queue_size = queue_size
        self.count = 0
        self.queue = deque() # Initialize two-sided queue

    # ADD: add sample, (state, action, reward, done_status, new_state), to queue
    def add(self, s, a, r, d, s_new):
        # s is current state, a is action taken
        # r is reward, d is if game ended (done)
        # s_new is next state
        experience = (s, a, r, d, s_new) # Make into tuple

        # Only add experience if queue NOT full
        if self.count < self.queue_size:
            self.queue.append(experience)
            self.count += 1
        else:
            # Otherwise, pop oldest experience and then add
            self.queue.popleft()
            self.queue.append(experience)
    

    # SIZE: just return # of experiences currenlty in queue
    def size(self):
        return self.count

    # SAMPLE: return a random batch of the queue of specified size
    def sample(self, batch_size):

        batch = [] # Initialize

        # Check if queue actually contains at least batch_size elements 
        if self.count < batch_size:
            batch = random.sample(self.queue, self.count) #NOT ENOUGH, so just randomly sample from current size
        else:
            batch = random.sample(self.queue, batch_size)

        # From batch, parse out and group states, actions, rewards, done's, and
        # s_news over all samples in batch
        # Then convert to list of numpy arrays
        # I.e. Given batch = [(s1, a1, ...), (s2, a2, ...), ...], 
        # This gives [(s1, s2,...), (a1, a2,...)
        # Then, store each in appropriate state, action, etc. batch variable
        s_batch, a_batch, r_batch, d_batch, s_new_batch = list(map(np.array, list(zip(*batch))))

        return s_batch, a_batch, r_batch, d_batch, s_new_batch
    
    # CLEAR: quickly clear experiences
    def clear(self):
        self.queue.clear()
        self.count = 0
