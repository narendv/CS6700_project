from config import *
import time
import numpy as np
import random

"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The methods "registered_reset_train" and "compute_action_train" are invoked here. 
Complete these functions to train your agent and save the state.

- Test Phase
The methods "registered_reset_test" and "compute_action_test" are invoked here. 
The final scoring is based on your agent's performance in this phase. 
Use the state saved in train phase here. 

"""


class Agent:
    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]
        if self.env_name == 'acrobot':
            self.state_size = self.config[0]
            self.action_size = self.config[1]
            self.W = np.random.randn(self.state_size,self.action_size)*1e-4
            self.prev_state = np.zeros((self.state_size,1))
            self.prev_act = 0
            self.discount_rate = 0.95
            self.learning_rate = 3*1e-5
            self.eps = 1
            self.reward_list = []
            self.grad_list =[]

        if self.env_name == 'taxi':
            self.state_size = self.config[0]
            # print("State size:", self.state_size)
            self.action_size = self.config[1]
            self.eps = 1
            self.discount_rate = 0.98
            self.learning_rate = 0.62
            self.Q = np.zeros([self.state_size, self.action_size])*1e-4
            self.prev_state = 0
            self.prev_act = 0

        if (self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc'):
            self.state_size = self.config[0]
            # print("State size:", self.state_size)
            self.action_size = self.config[1]
            self.eps = 1
            self.discount_rate = 1
            self.learning_rate = 0.5 if self.env_name=='kbca' else 0.8 if self.env_name=='kbcb' else 0.8
            self.get_qTableKBC()
            self.state_list =[]
            self.lam = 0.9
            self.prev_state = 0
            self.prev_act = 0
            self.t = 0
            # self.track = 0
            # self.stage = 0


    def get_qTableKBC(self):
        self.Q = np.zeros([self.state_size,self.action_size])
        self.Q[0][0] = 0
        for i in range(0, 16):
            self.Q[i+1][0] = 1000 * (2 ** i)

    def Q_update(self,obs,reward,done):
        q_next = np.zeros([self.action_size]) if done else self.Q[obs]
        q_target = reward + self.discount_rate * np.max(q_next)
        q_update = q_target - self.Q[self.prev_state, self.prev_act]
        self.Q[self.prev_state, self.prev_act] += self.learning_rate * q_update
        self.prev_state = obs
        return q_next


    def e_greedy(self,q_list,decay,done):
        p = random.random()
        if p<self.eps:
            action = random.choice(np.arange(self.action_size))
        else:
            action = np.argmax(q_list)
        if done:
            self.eps = self.eps * decay
        return action


    def et_greedy(self,prob,done):
        action = self.e_greedy(prob,1,done)
        if done:
            self.eps -= 1/2000
        return
    
    def softmax_grad(self,softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def softmax(self,x):
        return np.exp(x)/sum(np.exp(x))


    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        if self.env_name == 'acrobot':
            self.prev_act = np.random.randint(self.action_size)
            self.prev_state = obs
            self.reward_list = []
            self.grad_list = []
            self.t = 0

        if self.env_name == 'taxi':
            self.prev_act = random.choice(np.arange(self.action_size))
            self.prev_state = obs

        if (self.env_name == 'kbca' or self.env_name == 'kbcb'):
            state = obs.count(1)
            # self.prev_act = self.e_greedy(self.Q[state],1,False)
            self.prev_state = state
            self.prev_act = 1

        if self.env_name == 'kbcc':
            state = obs.count(1)
            action_greedy = np.argmax(self.Q[state])
            action_random = random.choice([1])
            q_action = action_random if random.random() < self.eps else action_greedy
            self.prev_state = state
            self.prev_act = q_action
        
        return self.prev_act

    def compute_action_train(self, obs, reward, done, info):
        """
        Use this function in the train phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        if self.env_name == 'acrobot':
            prob = self.softmax(obs.dot(self.W))
            self.prev_act = np.random.choice(self.action_size,p=prob)
            dsoftmax = self.softmax_grad(prob)[self.prev_act]
            dlog = dsoftmax/prob[self.prev_act]
            # grad = obs[None,:].T.dot(dlog[None,:])
            grad = np.outer(obs,dlog)
            self.reward_list.append(reward)
            self.grad_list.append(grad)
            self.prev_state = obs
            if done:
                for i in range(len(self.grad_list)):
                    self.W += self.learning_rate * self.grad_list[i] * sum(self.reward_list[i:])
        
        if self.env_name == 'taxi':
            q_next = self.Q_update(obs,reward,done)
            self.prev_act = self.e_greedy(q_next,0.85,done)

        if (self.env_name == 'kbca' or self.env_name =='kbcb'):
            next_state = obs.count(1)
            q_next = self.Q_update(next_state,reward,done)
            self.prev_act = 1

        if (self.env_name == 'kbcc'):
            next_state = obs.count(1)
            q_next = self.Q_update(next_state,reward,done)
            action_greedy = np.argmax(q_next[1:3])
            action_random = random.choice([1,2])
            self.prev_act = action_random if random.random() < self.eps else action_greedy
            if done:
                self.eps = self.eps * 0.999
        
        return self.prev_act
        

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        if self.env_name == 'acrobot':
            action = np.argmax(self.softmax(obs.dot(self.W)))

        if self.env_name == 'taxi':
            action = np.argmax(self.Q[obs])

        if (self.env_name == 'kbca' or self.env_name == 'kbcb' or  self.env_name == 'kbcc'):
            state = obs.count(1)
            action = np.argmax(self.Q[state])
        
        return action

    def compute_action_test(self, obs, reward, done, info):
        """
        Use this function in the test phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        if self.env_name == 'acrobot':
            action = np.argmax(self.softmax(obs.dot(self.W)))

        if (self.env_name == 'taxi'):
            action = np.argmax(self.Q[obs])

        if (self.env_name == 'kbca' or self.env_name == 'kbcb' or self.env_name == 'kbcc' ):
            state = obs.count(1)
            action= np.argmax(self.Q[state])
        
        return action
