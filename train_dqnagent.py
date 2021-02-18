from simulate_LOB import marketSimulation
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
ROUNDS = 5

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

simu = marketSimulation(intensities=intensity_values, n_ticks=N_TICKS, our_tick_idx=our_tick_idx, 
                          our_pos_idx=our_pos_idx, shares_left_idx=shares_left_idx, 
                          time_left_idx=time_left_idx, max_q=MAX_Q)

#%%


if __name__ == '__main__':
    for k in range(51):    
        decisons = ['limit', 'cancel', 'market']
        new_state = init_state_vec        
        decision, tick = simu.simulate_market(new_state)
        print('decision', decisons[decision])
        print('tick', tick)
        old_state = np.copy(new_state)
        new_state = simu.market_state_update(init_state_vec, decision, tick)
        print('Old state:', old_state)
        print('New state:', new_state)
        reward = simu.calc_reward(old_state, new_state, side='buy')
        print('Reward:', reward)

new_state = init_state_vec
for k in range(10):
   # Decide action
    action, q_table = agent.act(new_state)
    new_state = simu.simulate_market(new_state)
#    outcome_mem.append(outcome[0])
#    payoff = payoffs[0]
#    pay_history.append(payoff)
#    next_state = np.concatenate(outcome_mem)
#    next_state = np.array([next_state])
#    agent.remember(state, action_0, payoff, next_state)
#    state = next_state
#    #state_history.append(state)
#    print("Round: {round}/{rounds}, Score: {score}, Epsilon: {epsilon}"
#          .format(round=round+1,
#                  rounds=rounds,
#                  score=payoff,
#                  epsilon=agent.epsilon))
#    # train the agent with the experience of the round
#    if round > 32:
#        agent.replay(32)
#c_history = [q[0] for q in q_history]
#d_history = [q[1] for q in q_history]
#df = pd.DataFrame(
#    {
#        "Payoff":   pay_history,
#        "QCollude": c_history,
#        "QDefect":  d_history,
#        "Epsilon":  eps_history
#    }
#)
#if type(value) in [tuple, list]:
#    value = "{}-{}".format(value[0], value[1])
#df.index.name = "Round"
#df.to_csv("../Output/{0}_{1}_mem_{2}_rnds_{3}_{4}.csv"
#    .format(param, value, mem_len, rounds, tstamp),
#    columns=['Payoff','QCollude','QDefect','Epsilon'])
#return True
