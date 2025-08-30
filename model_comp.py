import numpy as np
import pandas as pd
import time
import warnings
import pickle
import os

from tools.parallel import *
from tools.fit_bms import * 
from tools.all_models import *
from tools.viz import viz 
viz.get_style()
## arrowleft=1
eps = np.finfo(float).eps
warnings.filterwarnings("ignore")
start = time.time()

##----------------------------input------------------------
basic_pth = "..\models\VG_vol"
models = ['RW', 'WSLS', 'transRW', 'Counter', 'PH', 'HGF', 'HMM0', 'HMM1', 'HMM2', 'BL1']
now_type = 'part1'
exp_cond = 'HC'
analyse_type = 'HC'
num_rows = 320
num_skips = 0

arg = Args(n_fit=1,n_sim=1,n_cores=2)

##-----------------------------fixed-------------------------
for model in models:
    now_agent = eval(model)
    agent = eval(model)
    
    actions = pd.read_csv(fr"{basic_pth}/{exp_cond}_actions.csv", header=None, nrows=num_rows, skiprows=num_skips)
    rewards = pd.read_csv(fr"{basic_pth}/{exp_cond}_rewards.csv", header=None, nrows=num_rows, skiprows=num_skips)
    resps = pd.read_csv(fr"{basic_pth}/{exp_cond}_actions.csv", header=None, nrows=num_rows, skiprows=num_skips)
    u_ts = pd.read_csv(fr"{basic_pth}/{exp_cond}_u_t.csv", header=None, nrows=num_rows, skiprows=num_skips)

    if __name__ == '__main__':
        pool = get_pool(args=arg)
        As = actions
        Rs = rewards  
        Resps = resps
        U_ts = u_ts
        # Fit the model
        results = []
        p_name = now_agent.p_name
        all_outputs = {}
        
        for i in range(As.shape[1]): 
            actions = (As[i]).squeeze()
            rewards = (Rs[i]).squeeze()
            resp = (Resps[i]).squeeze()
            u_t = (U_ts[i]).squeeze()
            
            data = preprocess(actions, rewards, resp, u_t)
            print("Finished preprocessing")  # 检查preprocess的完成情况

            res = fit_parallel(pool=pool, data=data,
                                agent = now_agent, p_name=p_name,
                                bnds=agent.bnds, pbnds=agent.pbnds, p_priors=None,
                                method='map', init=None, verbose= False, n_fits=20)
            
            print(f'Now finished subject{i} of {now_type}')
        
            output = {'subject': i}
            for idx, param_name in enumerate(p_name):
                output[param_name] = res['param'][idx]
            
            output.update({
                'log_like': res['log_like'],
                'AIC': res['aic'],
                'BIC': res['bic'],
                'H': res['H'],
                'H_inv': res['H_inv'],
                'log_post':res['log_post'],
                'n_param':res['n_param']
            })
            all_outputs[f'subject{i}'] = output
        pool.close()
        pool.join()         
        # Save all_outputs using pickle
        output_dir = os.path.join('mapoutput', f'{now_type}//{exp_cond}')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{model}_{analyse_type}output.pickle'), 'wb') as f:
            pickle.dump(all_outputs, f)
    ##-------------------------------------------------------------------                    