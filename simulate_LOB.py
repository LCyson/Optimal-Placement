import numpy as np

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



''' If BOTH_SIDES = True, first N_TICKS is buy side, next N_TICKS is sell side,
    otherwise, for now, if BOTH_SIDES = False, we use the buy side only '''


class marketSimulation:
    def __init__(self, intensities, n_ticks, our_tick_idx, our_pos_idx, 
                 shares_left_idx, time_left_idx, max_q=10):
        self.intensities = intensities
        self.n_ticks = n_ticks
        self.our_tick_idx = our_tick_idx
        self.our_pos_idx = our_pos_idx
        self.shares_left_idx = shares_left_idx
        self.time_left_idx = time_left_idx 
        self.max_q = max_q
    
    def agent_state_update(self, state, action):   
        if state[self.shares_left_idx] <= 0:
            #print('All shares have been sold.')
            return None 
        
        if state[self.time_left_idx] <= 0:
            #print('Out of time.')
            return None
       
        state[self.time_left_idx] -= 1               # subtract unit of time
        
        if action == 0:
            return state
        
        elif action == 1:
            state[state[self.our_tick_idx]] -= 1     # Decrease the queue size of current tick
            
            if state[self.our_tick_idx] == 0:
                state[self.shares_left_idx] -= 1     # We sell one share
                # We then move to the very last tick (this is just an arbitrary way to proceed)
                state[self.n_ticks-1] += 1
                state[self.our_tick_idx] = self.n_ticks 
            
            else:
                state[state[self.our_tick_idx]] += 1 # Increase the size of the next tick
                state[self.our_tick_idx] -= 1        # Move up a tick
        
        elif action == 2:
            
            if state[self.our_tick_idx] == self.n_ticks-1:
                return state 
            
            else:
                state[state[self.our_tick_idx]] -= 1 # Decrease the queue size of current tick
                
                state[self.our_tick_idx] += 1        # Move back a tick
                if state [state[self.our_tick_idx]] != self.max_q -1 :
                    state[state[self.our_tick_idx]] += 1 # Increase the size of the next tick
        
        elif action == 3:
            state[state[self.our_tick_idx]] -= 1     # Decrease the queue size of current tick
            state[self.shares_left_idx] -= 1         # We sell one share
            # We then move to the very last tick (this is just an arbitrary way to proceed)
            if state[self.n_ticks - 1] != self.max_q -1:
                state[self.n_ticks - 1] += 1
            state[self.our_tick_idx] = self.n_ticks
        
        return state
            
    def market_state_update(self, state, decision, tick):
        if state[self.shares_left_idx] <= 0:
            #print('All shares have been sold.')
            return None 
        
        if state[self.time_left_idx] <= 0:
            #print('Out of time.')
            return None
       
        state[self.time_left_idx] -= 1  # subtract unit of time
        if decision == 0:
            # decision 0 = limit order insertion
            state[tick] += 1
            if state[self.our_tick_idx] == tick:
                state[self.our_pos_idx] += 1
        if decision in (1, 2):
            # decision 1 = cancel order
            # decision 2 = market order 
            if state[tick] == 0:
                raise Exception('Trying to do market/cancel order on empty queue.')
            state[tick] -= 1
            if state[self.our_tick_idx] == tick:
                if state[self.our_pos_idx] != 0:
                    state[self.our_pos_idx] -= 1
                else:
                    state[self.shares_left_idx] -= 1         # We sell one share
                    # We then move to the very last tick (this is just an arbitrary way to proceed)
                    if state[self.n_ticks - 1] != self.max_q -1:
                        state[self.n_ticks - 1] += 1
                    state[self.our_tick_idx] = self.n_ticks
        return state
            
    def simulate_market(self, state, both_sides=True, check_valid=True):
        ''' For now, we are using the intensity values from Othmane's GitHub, but
            it appears to be for one tick on each side, namely, the best-bid and 
            best-ask prices. For now, I will assume that arrivals  to each of the 
            ticks is identical. '''
            
        if state is None:
            raise Exception('State is None')
        current_intensities = []
        # For each tick we find the intensity values of the states
        if both_sides:
            # First N_TICKS * 2 entries of state are the queue sizes
            queue_sizes = state[0: self.n_ticks * 2]
            #print(queue_sizes)
            for tick in range(self.n_ticks):
                    # add 1, since we (Python) start counting at 0, but intensities starts at 1
                    BB_size, BA_size = queue_sizes[tick] + 1, queue_sizes[-(tick + 1)] + 1 
                    #print(BB_size, BA_size)
                    intensity = list(self.intensities.loc[(BB_size, BA_size)])
                    current_intensities += intensity
            arrival_times = np.random.exponential(1/np.array(current_intensities))
            min_time_idx = arrival_times.argmin() 
            tick_idx = min_time_idx // 3
            decision_idx = min_time_idx % 3
            #print(queue_sizes)
            if (queue_sizes == np.zeros([1, self.n_ticks * 2])).all():
                decision_idx = 0
                tick_idx = np.random.randint(self.n_ticks * 2)
            
            # since the intensities are symmetric, I calulate the intensities for 
            # just one side of the book. Once we determine which tick we will have
            # an arrival, we then `flip a coin' to see whether its on the buy/sell side
            if (np.random.rand() > 0.5 and state[tick_idx + self.n_ticks] != 0) or state[tick_idx] == 0 or state[tick_idx] == self.max_q:
                #if state[tick_idx + n_ticks] == 0 and decision_idx:
                #    decision_idx = 0             cc
                #elif state[tick_idx + n_ticks] == max_q - 1:
                tick_idx = tick_idx + self.n_ticks  
            x = 0
    
            if check_valid:
                while((state[tick_idx] == 0 and decision_idx != 0) or (state[tick_idx] == self.max_q-1 and decision_idx == 0)): 
                    decision_idx, tick_idx = self.simulate_market(state, both_sides=True, check_valid=False)
                    x += 1
                    if x == 100:
                        break
            return decision_idx, tick_idx        
    
    def calc_reward(self, old_state, new_state, side='buy'):
        if side.lower() == 'buy':
            if new_state is None:
                reward = old_state[self.shares_left_idx] * -2
            elif old_state[self.shares_left_idx] - new_state[self.shares_left_idx] == 1:
                reward = self.n_ticks - new_state[self.our_tick_idx]
            else:
                reward = 0
        return reward


#if __name__ == '__main':
    #for k in range(51):    
        #decisons = ['limit', 'cancel', 'market']
        #new_state = init_state_vec        
        #decision, tick = simulate_market(new_state, intensity_values, 2, True)
        #print('decision', decisons[decision])
        #print('tick', tick)
        #old_state = np.copy(new_state)
        #new_state = market_state_update(init_state_vec, decision, tick)
        #print('Old state:', old_state)
        #print('New state:', new_state)
        #reward = calc_reward(old_state, new_state)
        #print('Reward:', reward)


    
    
    
                    
            