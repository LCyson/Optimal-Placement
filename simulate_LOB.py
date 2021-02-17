import os
import pandas as pd
import numpy as np
from agents import DQNAgent
from itertools import product
#%%
DATA_PATH = os.path.join(os.getcwd(), 'datalibrary', 'data')
N_TICKS = 2
BOTH_SIDES = True
TIME = 50
N_SHARES = 100
MAX_Q = 10

file_name = "intensity-values.csv"
intensity_values = pd.read_csv(os.path.join(DATA_PATH, file_name), index_col=[1,2])
intensity_values = intensity_values[intensity_values['Spread'] == 1]\
    .groupby(['BB size', 'BA size'])\
    .agg(dict(Limit='mean', Cancel='mean', Market='mean'))


tuples = list(product(range(1, MAX_Q + 1), repeat = 2))
intensity_values = intensity_values.loc[tuples]
intensity_values.loc[(1,1), ['Cancel', 'Market']] = 1e-6

max_queue_size = MAX_Q
#%%
# Initialize an agent, this uses Keras with TensorFlow as a backend
agent = DQNAgent(state_size=K * N_TICKS + 4, action_size=4)
#%%

''' Inputs:
       - q_-N, ..., q_-1,...,q_1,...,q_N the queue size for the first N ticks
       - Our tick position          (index = N_TICKS + 1)
       - Our position in the queue  (index = N_TICKS + 2)
       - Time left to place orders  (index = N_TICKS + 3)
       - N_SHARES - shares to place (index = N_TICKS + 4)
    Action:
       - (0) Nothing, stay in queue 
       - (1) Increase 1 tick, if at tick 1 - market buy
       - (2) Decrease 1 tick, if at last tick - stay in queue
       - (3) Market buy 
       
'''

# Start with a random state
# np.random.seed(420)
if BOTH_SIDES:
    K = 2
else:
    K = 1

''' If BOTH_SIDES = True, first N_TICKS is buy side, next N_TICKS is sell side,
    otherwise, for now, if BOTH_SIDES = False, we use the buy side only '''

init_queue_size_vec = np.random.randint(0, max_queue_size, K * N_TICKS)
init_tick_position = np.random.randint(0, K * N_TICKS) 
print(init_tick_position)
init_queue_position = np.random.randint(0, max(1,init_queue_size_vec[init_tick_position]))  


init_state_vec = np.append(init_queue_size_vec, init_tick_position)
init_state_vec = np.append(init_state_vec, init_queue_position)
init_state_vec = np.append(init_state_vec, TIME)
init_state_vec = np.append(init_state_vec, N_SHARES)

our_tick_idx = K * N_TICKS                  
our_pos_idx = K * N_TICKS + 1
time_left_idx = K * N_TICKS + 2
shares_left_idx = K * N_TICKS + 3

def agent_state_update(state, action, max_q=MAX_Q):   
    if state[shares_left_idx] <= 0:
        print('All shares have been sold.')
        return None 
    
    if state[time_left_idx] <= 0:
        print('Out of time.')
        return None
   
    state[time_left_idx] -= 1               # subtract unit of time
    
    if action == 0:
        return state
    
    elif action == 1:
        state[state[our_tick_idx]] -= 1     # Decrease the queue size of current tick
        
        if state[our_tick_idx] == 0:
            state[shares_left_idx] -= 1     # We sell one share
            # We then move to the very last tick (this is just an arbitrary way to proceed)
            state[N_TICKS-1] += 1
            state[our_tick_idx] = N_TICKS
        
        else:
            state[state[our_tick_idx]] += 1 # Increase the size of the next tick
            state[our_tick_idx] -= 1        # Move up a tick
    
    elif action == 2:
        
        if state[our_tick_idx] == N_TICKS-1:
            return state 
        
        else:
            state[state[our_tick_idx]] -= 1 # Decrease the queue size of current tick
            
            state[our_tick_idx] += 1        # Move back a tick
            if state [state[our_tick_idx]] != max_q -1 :
                state[state[our_tick_idx]] += 1 # Increase the size of the next tick
    
    elif action == 3:
        state[state[our_tick_idx]] -= 1     # Decrease the queue size of current tick
        state[shares_left_idx] -= 1         # We sell one share
        # We then move to the very last tick (this is just an arbitrary way to proceed)
        if state[N_TICKS-1] != max_q -1:
            state[N_TICKS-1] += 1
        state[our_tick_idx] = N_TICKS
    
    return state
        
def market_state_update(state, decision, tick, max_q=MAX_Q):
    if state[time_left_idx] <= 0:
        print('Out of time.')
        return None
   
    state[time_left_idx] -= 1  # subtract unit of time
    if decision == 0:
        # decision 0 = limit order insertion
        state[tick] += 1
        if state[our_tick_idx] == tick:
            state[our_pos_idx] += 1
    if decision in (1, 2):
        # decision 1 = cancel order
        # decision 2 = market order 
        if state[tick] == 0:
            raise Exception('Trying to do market/cancel order on empty queue.')
        state[tick] -= 1
        if state[our_tick_idx] == tick:
            if state[our_pos_idx] != 0:
                state[our_pos_idx] -= 1
            else:
                state[shares_left_idx] -= 1         # We sell one share
                # We then move to the very last tick (this is just an arbitrary way to proceed)
                if state[N_TICKS-1] != max_q -1:
                    state[N_TICKS-1] += 1
                state[our_tick_idx] = N_TICKS
    return state
        
def simulate_market(state, intensities, n_ticks, both_sides, max_q=MAX_Q, check_valid=True):
    ''' For now, we are using the intensity values from Othmane's GitHub, but
        it appears to be for one tick on each side, namely, the best-bid and 
        best-ask prices. For now, I will assume that arrivals  to each of the 
        ticks is identical. '''
    current_intensities = []
    # For each tick we find the intensity values of the states
    if both_sides:
        # First N_TICKS * 2 entries of state are the queue sizes
        queue_sizes = state[0: n_ticks * 2]
        print(queue_sizes)
        for tick in range(n_ticks):
                # add 1, since we (Python) start counting at 0, but intensities starts at 1
                BB_size, BA_size = queue_sizes[tick] + 1, queue_sizes[-(tick + 1)] + 1 
                print(BB_size, BA_size)
                intensity = list(intensity_values.loc[(BB_size, BA_size)])
                current_intensities += intensity
        arrival_times = np.random.exponential(1/np.array(current_intensities))
        min_time_idx = arrival_times.argmin() 
        tick_idx = min_time_idx // 3
        decision_idx = min_time_idx % 3
        
        
        if (queue_sizes == np.zeros([1, n_ticks * 2])).all():
            decision_idx = 0
            tick_idx = np.random.randint(n_ticks * 2 + 1)
        
        # since the intensities are symmetric, I calulate the intensities for 
        # just one side of the book. Once we determine which tick we will have
        # an arrival, we then `flip a coin' to see whether its on the buy/sell side
        if (np.random.rand() > 0.5 and state[tick_idx + n_ticks] != 0) or state[tick_idx] == 0 or state[tick_idx] == max_q:
            #if state[tick_idx + n_ticks] == 0 and decision_idx:
            #    decision_idx = 0             
            #elif state[tick_idx + n_ticks] == max_q - 1:
            tick_idx = tick_idx + n_ticks  
        x = 0

        if check_valid:
            while((state[tick_idx] == 0 and decision_idx != 0) or (state[tick_idx] == max_q-1 and decision_idx == 0)): 
                decision_idx, tick_idx = simulate_market(state, intensities, n_ticks, both_sides, max_q, False)
                x += 1
                if x == 100:
                    break
        return decision_idx, tick_idx        

def calc_reward(old_state, new_state, side='buy'):
    if side.lower() == 'buy':
        if new_state is None:
            reward = old_state[shares_left_idx] * -0.5 
        elif old_state[shares_left_idx] - new_state[shares_left_idx] == 1:
            reward = N_TICKS - new_state[our_tick_idx]
        else:
            reward = 0
    return reward

decisons = ['limit', 'cancel', 'market']
new_state = init_state_vec
for k in range(51):            
    decision, tick = simulate_market(new_state, intensity_values, N_TICKS, True)
    #print('decision', decisons[decision])
    #print('tick', tick)
    old_state = np.copy(new_state)
    new_state = market_state_update(init_state_vec, decision, tick)
    #print('Old state:', old_state)
    #print('New state:', new_state)
    reward = calc_reward(old_state, new_state)
    print('Reward:', reward)


    
    
    
                    
            