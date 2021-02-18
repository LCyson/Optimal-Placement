from simulate_LOB import marketSimulation
import pandas as pd
import numpy as np
import os
from itertools import product
from agents import DQNAgent

N_TICKS = 3
BOTH_SIDES = True
TIME = 100
N_SHARES = 5
MAX_Q = 10
ROUNDS = 5*10**6

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

def init_state():
    max_queue_size = MAX_Q
    init_queue_size_vec = np.random.randint(0, max_queue_size, K * N_TICKS)
    init_tick_position  = np.random.randint(0, N_TICKS) 
    init_queue_position = np.random.randint(0, max(1,init_queue_size_vec[init_tick_position]))  
    
    init_state_vec = np.append(init_queue_size_vec, init_tick_position)
    init_state_vec = np.append(init_state_vec, init_queue_position)
    init_state_vec = np.append(init_state_vec, TIME)
    init_state_vec = np.append(init_state_vec, N_SHARES)
    
    our_tick_idx = K * N_TICKS                  
    our_pos_idx = K * N_TICKS + 1
    time_left_idx = K * N_TICKS + 2
    shares_left_idx = K * N_TICKS + 3
    
    simu = marketSimulation(intensities=intensity_values, n_ticks=N_TICKS, our_tick_idx=our_tick_idx, 
                              our_pos_idx=our_pos_idx, shares_left_idx=shares_left_idx, 
                              time_left_idx=time_left_idx, max_q=MAX_Q)

    return init_state_vec, simu

new_state, simu = init_state()
#%%


#if __name__ == '__main__':
#    for k in range(51):    
#        decisons = ['limit', 'cancel', 'market']
#        #new_state = init_state_vec        
#        decision, tick = simu.simulate_market(new_state)
#        print('decision', decisons[decision])
#        print('tick', tick)
#        old_state = np.copy(new_state)
#        new_state = simu.market_state_update(new_state, decision, tick)
#        print('Old state:', old_state)
#        print('New state:', new_state)
#        reward = simu.calc_reward(old_state, new_state, side='buy')
#        print('Reward:', reward)

actions = ['stay', 'up', 'down', 'market']
for k in range(ROUNDS):
   # Decide action

   if new_state is None:
       new_state, simu = init_state()
   if k % 1000 == 0 and k > 0:
       print("round", k)
       print("state:", new_state)
       print("Q-table:")
       print(agent.model.predict(np.array([new_state])))       
   old_state = np.copy(new_state)
   action, q_table = agent.act(np.array([new_state]))
   new_state = simu.agent_state_update(new_state, action)
   reward = simu.calc_reward(old_state, new_state, side='buy')
   if old_state is not None and new_state is not None:
       agent.remember(old_state, action, reward, new_state)
       decision, tick = simu.simulate_market(new_state)
       if decision is not None:
           new_state = simu.market_state_update(new_state, decision, tick)
           reward = simu.calc_reward(old_state, new_state, side='buy')
           if old_state is not None and new_state is not None:
               agent.remember(old_state, action, reward, new_state)
# agent learns from old states by remembering and replaying them
   if k > 32:
       agent.replay(32)
