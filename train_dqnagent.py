import simulate_LOB as simu
import pandas as pd
import numpy as np
import os
from itertools import product
from agents import DQNAgent

N_TICKS = 2
BOTH_SIDES = True
TIME = 50
N_SHARES = 100
MAX_Q = 10
ROUNDS = 500

# Start with a random state
# np.random.seed(420)
if BOTH_SIDES:
    K = 2
else:
    K = 1

#%%
# Initialize an agent, this uses Keras with TensorFlow as a backend
if 'agent' not in locals() and 'agent' not in globals():    # So I'm not needlessly creating new TF connections while debugging
    agent = DQNAgent(state_size=K * N_TICKS + 4, action_size=4)
    
#%%
DATA_PATH = os.path.join(os.getcwd(), 'datalibrary', 'data')


file_name = "intensity-values.csv"
intensity_values = pd.read_csv(os.path.join(DATA_PATH, file_name), index_col=[1,2])
intensity_values = intensity_values[intensity_values['Spread'] == 1]\
    .groupby(['BB size', 'BA size'])\
    .agg(dict(Limit='mean', Cancel='mean', Market='mean'))


tuples = list(product(range(1, 10 + 1), repeat = 2))
intensity_values = intensity_values.loc[tuples]
intensity_values.loc[(1,1), ['Cancel', 'Market']] = 1e-6
#%%
max_queue_size = MAX_Q
init_queue_size_vec = np.random.randint(0, max_queue_size, K * N_TICKS)
init_tick_position  = np.random.randint(0, K * N_TICKS) 
init_queue_position = np.random.randint(0, max(1,init_queue_size_vec[init_tick_position]))  

init_state_vec = np.append(init_queue_size_vec, init_tick_position)
init_state_vec = np.append(init_state_vec, init_queue_position)
init_state_vec = np.append(init_state_vec, TIME)
init_state_vec = np.append(init_state_vec, N_SHARES)

our_tick_idx = K * N_TICKS                  
our_pos_idx = K * N_TICKS + 1
time_left_idx = K * N_TICKS + 2
shares_left_idx = K * N_TICKS + 3


if __name__ == '__main__':
    for k in range(51):    
        decisons = ['limit', 'cancel', 'market']
        new_state = init_state_vec        
        decision, tick = simu.simulate_market(new_state, intensity_values, 2, True)
        print('decision', decisons[decision])
        print('tick', tick)
        old_state = np.copy(new_state)
        new_state = simu.market_state_update(init_state_vec, decision, tick, 
                                             n_ticks=N_TICKS,
                                             our_tick_idx=our_tick_idx, 
                                             our_pos_idx=our_pos_idx, 
                                             shares_left_idx=shares_left_idx, 
                                             time_left_idx=time_left_idx, 
                                             max_q=MAX_Q)
        print('Old state:', old_state)
        print('New state:', new_state)
        reward = simu.calc_reward(old_state, new_state, n_ticks=N_TICKS, 
                                  our_tick_idx=our_tick_idx, 
                                  shares_left_idx=shares_left_idx, 
                                  time_left_idx=time_left_idx, side='buy')
        print('Reward:', reward)
