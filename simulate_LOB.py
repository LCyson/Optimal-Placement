import numpy as np

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
        self.market_flag = 0
    

    def sell_share(self, state):
        if state[self.shares_left_idx] > 0:
            state[self.shares_left_idx] -=1
        else:
            raise Exception('Error')
        return state
    
    def increment_current_queue(self, state):
        if state[state[self.our_tick_idx]] < self.max_q - 1:
            state[state[self.our_tick_idx]] +=1 
        return state
                   
    def decriment_current_queue(self, state):
        if state[state[self.our_tick_idx]] > 0:
            state[state[self.our_tick_idx]] -= 1
        return state 
    
    def reset_position(self, state):
        # If not at first tick:
        if state[self.our_tick_idx] < self.n_ticks - 1:
            state = self.decriment_current_queue(state)
            state[self.our_tick_idx] = self.n_ticks - 1
            state = self.increment_current_queue(state)
        # set position to length of queue (i.e. put us last in queue)
        state[self.our_pos_idx] = state[self.n_ticks - 1]
        return state    
    
    def increase_tick_pos(self, state):
        if state[self.our_tick_idx] < self.n_ticks - 1:
            state[self.our_tick_idx] += 1
            state[self.our_pos_idx] = state[state[self.our_tick_idx]]
        else:
            self.market_flag = 1
            state = self.sell_share(state)
            state = self.reset_position(state)
        return state
    
    def decrease_tick_pos(self, state):
        if state[self.our_tick_idx] > 0:
            state[self.our_tick_idx] -= 1
        state[self.our_pos_idx] = state[state[self.our_tick_idx]]
        return state
        
    def agent_state_update(self, state, action): 
    
        '''
        Parameters
        ----------
        state : array 
            The current market state
        action : int 
            The agent's choice :
                0=do nothing
                1=move up a tick (market sell if at first tick)
                2=move back a tick (do nothing if at last tick)
                3=market order
        Returns
        -------
        state : array or None
            the new state after the agent's decision
        '''
        
        # if the number of shares left is <= 0 return None
        if state[self.shares_left_idx] <= 0:
            return None 
        
        # if the time left is <= retrun None
        if state[self.time_left_idx] <= 0:
            return None
        
        # subtract unit of time
        state[self.time_left_idx] -= 1              
        
        # If action is 0, we do nothing, return the same state
        if action == 0:
            pass
        
        if action == 1:
            state = self.increase_tick_pos(state)

        
        elif action == 2:
            state = self.decrease_tick_pos(state)
        
        elif action == 3:
            self.market_flag = 1
            state = self.sell_share(state)
            state = self.reset_position(state)
            
        
        return state
            
    def market_state_update(self, state, decision, tick):
        if state[self.shares_left_idx] <= 0:
            return None 
        
        if state[self.time_left_idx] <= 0:
            return None
        
        # subtract unit of time
        state[self.time_left_idx] -= 1  
        
        if decision == 0: # limit order insertion
            if state[tick] < self.max_q - 1:
                state[tick] += 1
            if state[self.our_tick_idx] == tick:
                if state[self.our_pos_idx] < self.max_q - 1:
                    state[self.our_pos_idx] += 1
                    
        if decision == 1: # cancel
            # if the cancel is at our tick
            if state[self.our_tick_idx] == tick:
                # if we are the only one are queue, error
                # this is probably going to cause an error becuase we can still get cancel orders if 1 in queue
                if state[tick] <= 1:
                    raise Exception('Market trying to cancel our order')
                
                # otherwise, decrease size of queue/our position    
                elif state[self.our_pos_idx] > 1:
                    state[tick] -= 1
                    state[self.our_pos_idx] -= 1
                else:
                    state[tick] -= 1 # canceled order is behind ours in queue
                    # eventually this should be random based on uniform dist or something
                    
            # if the cancel is not at our tick        
            else:
                if state[tick] <= 0:
                    raise Exception('Market trying to cancel order from empty queue')
                else:
                    state[tick] -= 1
        
        if decision == 2: # market order
            # if the market order is at our tick
            if state[self.our_tick_idx] == tick:
                if state[self.our_pos_idx] == 0:
                    self.market_flag = 0
                    state = self.sell_share(state)
                    state = self.reset_position(state)
                else:
                    state[tick] -= 1
                    state[self.our_pos_idx] -= 1
            else:
                if state[tick] <= 0:
                    raise Exception ("Merket order at empty queue")
                else:
                    state[tick] -= 1
        
        return state
            
    def simulate_market(self, state, both_sides=True, check_valid=True):
        ''' For now, we are using the intensity values from Othmane's GitHub, but
            it appears to be for one tick on each side, namely, the best-bid and 
            best-ask prices. For now, I will assume that arrivals  to each of the 
            ticks is identical. '''
        if (state[0:self.n_ticks * 2] < 0).any():
            return 0, np.argmin(state)
        if state is None:
            raise Exception('State is None')
        current_intensities = []
        # For each tick we find the intensity values of the states
        if both_sides:
            # First N_TICKS * 2 entries of state are the queue sizes
            queue_sizes = state[0: self.n_ticks * 2]
            for tick in range(self.n_ticks):
                    # add 1, since we (Python) start counting at 0, but intensities starts at 1
                    BB_size, BA_size = queue_sizes[tick] + 1, queue_sizes[-(tick + 1)] + 1 
                    intensity = np.array(self.intensities.loc[(BB_size, BA_size)]) \
                                            * 0.95 ** (self.n_ticks - tick - 1)
                    current_intensities = np.append(current_intensities, intensity)
            arrival_times = np.random.exponential(1/current_intensities)
            min_time_idx = arrival_times.argmin() 
            tick_idx = min_time_idx // 3
            decision_idx = min_time_idx % 3
            if (queue_sizes == np.zeros([1, self.n_ticks * 2])).all():
                decision_idx = 0
                tick_idx = np.random.randint(self.n_ticks)        
            ''' since the intensities are symmetric, I calulate the intensities for 
                just one side of the book. Once we determine which tick we will have
                an arrival, we then `flip a coin' to see whether its on the buy/sell side '''
            if (np.random.rand() > 0.5 and state[tick_idx + self.n_ticks] != 0) \
                or state[tick_idx] == 0 or state[tick_idx] == self.max_q - 1:
                tick_idx = tick_idx + self.n_ticks 
            x = 0
    
            if check_valid:
                while((state[tick_idx] == 0 and decision_idx != 0) \
                       or (tick_idx == state[self.our_tick_idx] \
                       and state[tick_idx] == 1 and decision_idx == 1) \
                       or (state[tick_idx] == self.max_q-1 and decision_idx == 0)): 
                    decision_idx, tick_idx = self.simulate_market(
                                                state, 
                                                both_sides=True, 
                                                check_valid=False)
                    x += 1
                    if x == 1000:
                        print(state)
                        raise Exception("error")
            return decision_idx, tick_idx        
    
    def calc_reward(self, old_state, new_state, side='buy'):
        if side.lower() == 'buy':
            if new_state is None:
                reward = old_state[self.shares_left_idx] * -1.2
            elif old_state[self.shares_left_idx] - new_state[self.shares_left_idx] == 1 and self.market_flag == 0:
                reward = self.n_ticks - old_state[self.our_tick_idx]
            elif old_state[self.shares_left_idx] - new_state[self.shares_left_idx] == 1:
                reward = -1
            else:
                reward = -.01
        return reward
     
if __name__ == "__main__":
    # For testing 
    from agents import DQNAgent
    from train_dqnagent import init_state
    N_TICKS = 4
    BOTH_SIDES = True
    TIME = 100
    N_SHARES = 5
    MAX_Q = 10
    ROUNDS = 5*10**5
    
    # Start with a random state
    # np.random.seed(420)
    if BOTH_SIDES:
        K = 2
    else:
        K = 1
        
    # So I'm not needlessly creating new TF connections while debugging
    if 'agent' not in locals() and 'agent' not in globals():   
        # Initialize an agent, this uses Keras with TensorFlow as a backend
        agent = DQNAgent(state_size=K * N_TICKS + 4, action_size=4)
        
    r = 0
    actions = ['stay', 'up', 'down', 'market']
    for k in range(ROUNDS):
        new_state, simu = init_state()
        if k % 100 == 0 and k > 0:
            print("----------------------------------------------------------")
            print("round", k)
            print("state:", new_state)
            print("Q-table:")
            print(agent.model.predict(np.array([new_state]))[0])    
            print("----------------------------------------------------------")
       
        j = 0
        while new_state is not None: 
            j += 1
            if j % 10 == 0 and k % 1000 == 0 and j > 0 and k > 0:
                print("----------------------------------------------------------")
                print("round", k)
                print("episode", j)
                print("state:", new_state)
                print("Q-table:")
                print(agent.model.predict(np.array([new_state]))[0])  
                print("----------------------------------------------------------")
            old_state = np.copy(new_state)
            action, q_table = agent.act(np.array([new_state]))
            new_state = simu.agent_state_update(new_state, action)
            reward = simu.calc_reward(old_state, new_state, side='buy')
            if old_state is not None and new_state is not None and j > 1:
                agent.remember(old_state, action, reward, new_state)
                r += 1
                decision, tick = simu.simulate_market(new_state)
                if decision is not None:
                    new_state = simu.market_state_update(new_state, decision, tick)
                    reward = simu.calc_reward(old_state, new_state, side='buy')
                    if old_state is not None and new_state is not None:
                        agent.remember(old_state, action, reward, new_state)
                        r += 1
        # agent learns from old states by remembering and replaying them
        if r > 32:
            agent.replay(32)