import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        #return ''.join([
        #    self.get(self.frog_x - 1, self.frog_y - 1) or '_',
        #    self.get(self.frog_x, self.frog_y - 1) or '_',
        #    self.get(self.frog_x + 1, self.frog_y - 1) or '_',
        #])

        
        #Updated key computing method that checks a 3x3 grid around the frog and combines it into a key string
        return ''.join([
            #Get character top left
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            #Get character right above
            self.get(self.frog_x, self.frog_y - 1) or '_',
            #Get character top right
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
            #Get character to left
            self.get(self.frog_x - 1, self.frog_y) or '_',
            #Get position
            self.get(self.frog_x, self.frog_y) or '_',
            #Get to the right
            self.get(self.frog_x + 1, self.frog_y) or '_',
            #Get bottom left
            self.get(self.frog_x - 1, self.frog_y + 1) or '_',
            #Get right below
            self.get(self.frog_x, self.frog_y + 1) or '_',
            #Get bottom right
            self.get(self.frog_x + 1, self.frog_y + 1) or '_',
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):

        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'
        
        #Initializers for Q-Learning algorithm, alpha(learning rate), gamma(discount factor), epsilon(exploration rate)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.075

        #Initializers for prev state and action
        self.previous_action = None
        self.previous_state = None

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        print(f"Q-table saved to {self.path}")
        return self

    def choose_action(self, state_string):
        
        #Construct internal q_statse using given state string
        state = Q_State(state_string)
        state_key = state.key

        #If the key associated with the state is not within the current q-table, add the key to the q_table, and adds the the initial starting value of 0.0 as the value
        if state_key not in self.q:
            self.q[state.key] = {action: 0.0 for action in State.ACTIONS}

        #Epsilon-greedy search, if random num between 0 and 1 is less the epsilon value(0.1), select a random action. else, choose the action with the max Q value
        if random.uniform(0,1) < self.epsilon:
            action = random.choice(State.ACTIONS)
        else:
            action = max(self.q[state_key], key = self.q[state.key].get)
            
            
        #If training is on, update the q_table, save q_table after each action
        if self.train and self.previous_state is not None and self.previous_action is not None:
            self.update(self.previous_state, state_string)
            self.save()
        
        self.previous_state = state_key
        self.previous_action = action


        return action

    #Base Q_New value equation came from the equation given in class, tweaked around with alpha, epsilon and gamma values
    def update(self, previous_state, current_state_str):
        #Initialize reward value and new state key
        current_state = Q_State(current_state_str)
        reward = current_state.reward()
        next_state_key = current_state.key

        #If the next_state's key is not in the current q_table, add the key to the q_table and add initial starting value of 0.0 as the value
        if next_state_key not in self.q:
            self.q[next_state_key] = {act: 0.0 for act in State.ACTIONS}

        #Iniitalize old value with previous state key and previous action value
        #Initialize observed reward by taking the max value
        old_val = self.q[self.previous_state][self.previous_action]
        observed_reward = max(self.q[next_state_key].values())
        
        #Initalize new Q value based on equation given in class, replacing the initial 0.0 value
        q_new = ((1 - self.alpha) * old_val) + (self.alpha * (reward + self.gamma * observed_reward))


        self.q[self.previous_state][self.previous_action] = q_new
