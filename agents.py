import keras
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 memory=50,
                 gamma=.99,
                 epsilon=.60,
                 epsilon_min=0.001,
                 epsilon_decay=0.99999,
                 lr=0.0001,
                 neurons=(40,40),
                 activation=('relu', 'relu'),
                 ):
        from keras import backend as K
        import tensorflow as tf
        #K.clear_session()
        self.neurons = neurons 
        self.state_size = state_size       
        self.action_size = action_size     
        self.memory = deque(maxlen=memory) 
        self.gamma = gamma                 
        self.epsilon = epsilon             
        self.epsilon_min = epsilon_min     
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.tf_session = K.get_session()  # this creates a new session
        self.tf_graph = tf.get_default_graph()
        self.model = self._build_model(neurons, activation)

    def _build_model(self, neurons, activation):
        from keras.models import Sequential
        from keras.layers import Dense, Activation, InputLayer
        from keras.optimizers import Adam

        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                # Neural Net for Deep-Q learning Model
                model = Sequential()
                model.add(Dense(               # Add a hidden layer
                    neurons[0],                # Number of Outputs
                    input_dim=self.state_size, # Number of inputs
                    activation=activation[0])  # Activation method
                )
                ''' Note: I've tried sigmoid and relu, relu is definitely faster '''
                model.add(Dense(
                    neurons[1],
                    activation=activation[1])
                )
                model.add(Dense(
                    self.action_size,
                    activation='linear')
                )
                model.compile(
                    loss='mse',
                    optimizer=Adam(lr=self.learning_rate)
                )
                return model

    def remember(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])

    def act(self, state):
        ''' This funtion decides whether to act randomly or use the NN to
            predict the best action '''
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.action_size), ['NA','NA']]
        ''' act_value is a vector of length of action space. It's essentially the row of the
            Q-table corresponding to the current chain of states. '''
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                act_values = self.model.predict(state)
        #print("Q-table:")
        #print(act_values[0])
        return [np.argmax(act_values[0]), act_values[0]]  # returns action and Q-table

    def replay(self, batch_size):
        ''' This function uses the memory of outcomes and rewards to train the
            model. batch_size = the size of the sample we draw from memory '''
        minibatch = random.sample(self.memory, batch_size)
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                for state, action, reward, next_state in minibatch:
                    #print("next_state", next_state)
                    #print("next state shape", np.array([next_state][0]).shape)
                    #print("next state shape", np.array([next_state]).shape)
                    target = reward + self.gamma * \
                        np.amax(self.model.predict(np.array([next_state])))
                    target_f = self.model.predict(np.array([state]))
                    target_f[0][action] = target
                    self.model.fit(
                        np.array([state]),
                        target_f,
                        epochs=1,
                        verbose=0,
                        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        self.model.save(path)
