import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), 'datalibrary', 'data')

file_name = "intensity-values.csv"
intensity_values = pd.read_csv(os.path.join(DATA_PATH, file_name), index_col=0)
intensity_values = intensity_values[intensity_values['Spread'] == 1]\
    .groupby(['BB size'])\
    .agg(dict(Limit='mean', Cancel='mean', Market='mean'))

intensity_values.reset_index(inplace = True)
intensity_values.loc[0, ['Cancel','Market']] = 0
size_q = intensity_values.shape[0]

# simulation parameters:
np.random.seed(5150)
qb_0 = 2                                    
bb_pos_0 = qb_0 -1
lob_state_0 = [qb_0, bb_pos_0] 
write_option = True
nb_iter = 100
gamma = 0.1     

h_0_init = 5 * np.ones((size_q, size_q + 1))    # 
h_0_stay_init = np.ones((size_q, size_q + 1))
h_0_mkt_init = np.ones((size_q, size_q + 1))

h_0 = np.array(h_0_init)
h_0_Stay = np.array(h_0_stay_init)
h_0_Mkt = np.array(h_0_mkt_init)

#h_0_theo = op_place_1_n_2.Read_h_0_theo(df_bis["Value_opt"].values,size_q,reward_exec_1)
#Error = lambda x : op_place_1_n_2.error_1(np.nan_to_num(h_0_theo),np.nan_to_num(x))



def reward_exec(qsame, bb_pos, gain = 2, cost_out = -1, cost_stay = -0.5):
    if bb_pos ==  0: ## win if execution
        return gain
    elif bb_pos ==  -1: ## cost of a market order
        return cost_out
    else : ## cost of waiting
        return cost_stay

#### Function that encodes the new order book state after a market order
def state_after_market(lob_state_0, reward_0, reward_exec=reward_exec):
    lob_state_res = list(lob_state_0)
    reward_res = 0 # float(reward_0)
    h_0_market = 0
    if (lob_state_res[1] >= 1): 
        ### Regeneration 
        lob_state_res[1] = -1
        lob_state_res[0] =  max(lob_state_res[0] - 1, 0)
        reward_res = reward_exec(lob_state_res[0],lob_state_res[1])
    return [lob_state_res, reward_res, h_0_market]