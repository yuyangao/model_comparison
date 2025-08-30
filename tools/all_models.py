import numpy as np
import pandas as pd
import time
import warnings

from tools.parallel import *
from tools.fit_bms import * 
from scipy.stats import beta, bernoulli, norm, uniform  
from tools.viz import viz 
viz.get_style()
## arrowleft=1
eps = np.finfo(float).eps
warnings.filterwarnings("ignore")
start = time.time()

##----------------------------input------------------------

def clip_exp(x):
    x = np.clip(x, a_min=-1e15, a_max=50)
    return np.exp(x)

def sigmoid(beta, A, B):
    p_act = 1 / (1 + clip_exp(beta * (B - A)))
    return p_act

def trans(x):
    p_param = 1/(1+clip_exp(-x))
    return p_param

def rbeta(r, v):
    '''
    reparameterise beta distribution
    r = a/a+b
    v = -log(a+b)
    '''
    a = r*np.exp(-v)
    b = (1-r)*np.exp(-v)
    return beta(a, b)
    
#all distributions
alpha_dist = uniform(-2,1)#uniform(0,1)# distribution of learning rate 
beta_dist = uniform(-2,2)#uniform(0,50)# distribution of inverse temperature
gamma_dist = uniform(-2,2)
c_dist = uniform(-1,1)
d_dist = uniform(-1,1)
k_dist = uniform(-2,2)
eta_dist = uniform(-2,2)
kappa_dist = uniform(-2,2)
omega2_dist = uniform(-10,20)
omega3_dist = uniform(-10,20)
Beta_dist = uniform(-2,2)
Gamma_dist = uniform(-10,20)
inputs = []

def preprocess(actions, rewards, resp, u_t):
    df = pd.DataFrame({'actions':actions, 'rewards':rewards, 'resp':resp, 'u_t':u_t})
    return df

class Chance:
    p_name = ['beta']
    bnds = [(-10, 10)]
    pbnds = [(-5, 5)]
    prior = [norm(0,10)]

    def __init__(self, param):
        self.beta = clip_exp(param[0])        
    
    def FIT(self, df):
        V_L, V_R, neg_lok = 0, 0, 0
       
        for t in range(len(df['actions'])):
            P_L = sigmoid(self.beta, V_L, V_R)
            V_L, V_R = 0.5, 0.5        
            P = P_L if t == 0 else P_L if df.at[t, 'actions'] == 1 else 1 - P_L
            neg_lok += -np.log(P + eps)
        
        return neg_lok

class WSLS:
    p_name = ['beta']
    bnds = [(-10, 10)]
    pbnds = [(-5, 5)]
    prior = [norm(0,10)]

    def __init__(self, param):
        self.beta = clip_exp(param[0])        
    
    def FIT(self, df):
        V_L, V_R, neg_lok = 0, 0, 0
       
        for t in range(len(df['actions'])):
            P_L = sigmoid(self.beta, V_L, V_R)
            
            if t > 0:
                if df.at[t, 'actions'] == 1:
                    V_L, V_R = (1.0, -1.0) if df.at[t, 'rewards'] == 1 else (-1.0, 1.0)
                else:
                    V_L, V_R = (-1.0, 1.0) if df.at[t, 'rewards'] == 1 else (1.0, -1.0)
            
            if t == (len(df['actions'])) // 2:
                V_L, V_R= 0, 0

            P = P_L if t == 0 else P_L if df.at[t, 'actions'] == 1 else 1 - P_L
            neg_lok += -np.log(P + eps)
        
        return neg_lok
    #x0 = [beta_dist.rvs()]

class RW:
    '''
    standard Rescorla-Wagner 
    
    update function:
        Vt = Vt-1 + α * PE(t-1)
        PE(t-1) = Rt-1  -  Vt-1 
    '''
    p_name = ['alpha', 'beta']
    bnds = [(0, 1), (-10, 10)]
    pbnds = [(0, 0.5), (-5, 5)]
    prior = [norm(0,1.55), norm(0,10)]
    
    def __init__(self, param) :
        self.alpha = trans(param[0])
        self.beta = clip_exp(param[1])
    
    def FIT(self, df):
        '''
        dataframe
        cols: actions、rewards
        '''
        V_L, V_R, neg_lok = 0, 0, 0

        for t in range(len(df['actions'])):
            # sigmoid 
            P_L = 1 / (1 + np.exp(self.beta * (V_R - V_L)))
            ##----------------------------------##
            ##   Vt = Vt-1 + α * PE(t-1)        ##
            ##   PE(t-1) = Rt-1  -  Vt-1        ##
            ##----------------------------------##
            if t == 0 or t==(len(df['actions'])) // 2:
                P = 0.5  
            else:
                if df.at[t, 'actions'] == 1:
                    PE_L = df.at[t - 1, 'rewards'] - V_L
                    V_L = V_L + self.alpha * PE_L
                    P = P_L
                else:
                    PE_R = df.at[t - 1, 'rewards'] - V_R
                    V_R = V_R + self.alpha * PE_R
                    P = 1 - P_L
            
            neg_lok += -np.log(P + eps)
            
        return neg_lok

class RW_vol:
    '''
    standard Rescorla-Wagner 
    
    update function:
        Vt = Vt-1 + α * PE(t-1)
        PE(t-1) = Rt-1  -  Vt-1 
    '''
    p_name = ['stab_alpha','vol_alpha', 'beta']
    bnds = [(0, 1), (0, 1), (-10, 10)]
    pbnds = [(0, 0.5), (0, 0.5), (-5, 5)]
    prior = [norm(0,1.55), norm(0,1.55), norm(0,10)]
    
    def __init__(self, param) :
        self.stab_alpha = trans(param[0])
        self.vol_alpha = trans(param[1])
        self.beta = clip_exp(param[2])
    
    def FIT(self, df):

        V_L, V_R, neg_lok = 0, 0, 0

        for t in range(len(df['actions'])):
            P_L = 1 / (1 + np.exp(self.beta * (V_R - V_L)))
            ##----------------------------------##
            ##   Vt = Vt-1 + α * PE(t-1)        ##
            ##   PE(t-1) = Rt-1  -  Vt-1        ##
            ##----------------------------------##
            if t == 0 or t==(len(df['actions'])) // 2:
                P = 0.5  
            else:
                if df.at[t, 'actions'] == 1:
                    PE_L = df.at[t - 1, 'rewards'] - V_L
                    if t <= (len(df['actions'])) // 2:
                        V_L = V_L + self.stab_alpha * PE_L
                    else:
                        V_L = V_L + self.vol_alpha * PE_L
                    P = P_L
                else:
                    PE_R = df.at[t - 1, 'rewards'] - V_R
                    if t <= (len(df['actions'])) // 2:
                        V_R = V_R + self.stab_alpha * PE_R
                    else:
                        V_R = V_R + self.vol_alpha * PE_R
                    P = 1 - P_L
            
            neg_lok += -np.log(P + eps)
            
        return neg_lok

class RW_BL:
    '''
    standard Rescorla-Wagner 
    
    update function:
        Vt = Vt-1 + α * PE(t-1)
        PE(t-1) = Rt-1  -  Vt-1 
    '''
    p_name = ['stab_alpha1','vol_alpha1','stab_alpha2', 'vol_alpha2','beta','Gamma']
    bnds = [(0, 1), (0, 1), (0, 1), (0, 1), (-10, 10),(-20, 20)]
    pbnds = [(0, 0.5), (0, 0.5),(0, 0.5), (0, 0.5), (-5, 5), (-5, 5)]
    prior = [norm(0,1.55), norm(0,1.55),norm(0,1.55), norm(0,1.55), norm(0,10), norm(0,10)]
    
    def __init__(self, param) :
        self.stab_alpha1 = trans(param[0])
        self.vol_alpha1 = trans(param[1])
        self.stab_alpha2 = trans(param[2])
        self.vol_alpha2 = trans(param[3])        
        self.beta = clip_exp(param[4])
        self.Gamma = param[5]

    def F_cal(self, gamma, r):
        F = np.max([np.min([(gamma *(r - 0.5) + 0.5), 1]), 0])
        return F
    
    def FIT(self, df):

        V_L, V_R, neg_lok = 0, 0, 0

        for t in range(len(df['actions'])):
            P_L = 1 / (1 + np.exp(self.beta * (V_R - V_L)))
            ##----------------------------------##
            ##   Vt = Vt-1 + α * PE(t-1)        ##
            ##   PE(t-1) = Rt-1  -  Vt-1        ##
            ##----------------------------------##  
            if t <= (len(df['actions'])) // 4:
                lr = self.stab_alpha1
            elif t > (len(df['actions'])) // 4 and t <= (len(df['actions'])) // 2:   
                lr = self.vol_alpha1
            elif t <= (len(df['actions'])) // (4/3) and t > (len(df['actions'])) // 2: 
                lr = self.stab_alpha2
            elif t > (len(df['actions'])) // (4/3):
                lr = self.vol_alpha2

            if t == 0 or t==(len(df['actions'])) // 2:
                P = 0.5            
            else:
                if df.at[t, 'actions'] == 1:
                    PE_L = df.at[t - 1, 'rewards'] - V_L
                    delta_L = V_L + lr * PE_L
                    V_L = self.F_cal(self.Gamma, delta_L)
                    V_R = self.F_cal(self.Gamma, (1 - V_L))   
                    P = P_L
                else:
                    PE_R = df.at[t - 1, 'rewards'] - V_R
                    delta_R = V_R + lr * PE_R    
                    V_R = self.F_cal(self.Gamma, delta_R)
                    V_L = self.F_cal(self.Gamma, (1 - V_R)) 
                    P = 1 - P_L
            
            neg_lok += -np.log(P + eps)
            
        return neg_lok

class RW_vol2:
    '''
    standard Rescorla-Wagner 
    
    update function:
        Vt = Vt-1 + α * PE(t-1)
        PE(t-1) = Rt-1  -  Vt-1 
    '''
    p_name = ['stab_alpha1','vol_alpha1','stab_alpha2', 'vol_alpha2','beta']
    bnds = [(0, 1), (0, 1), (0, 1), (0, 1), (-10, 10)]
    pbnds = [(0, 0.5), (0, 0.5),(0, 0.5), (0, 0.5), (-5, 5)]
    prior = [norm(0,1.55), norm(0,1.55),norm(0,1.55), norm(0,1.55), norm(0,10)]
    
    def __init__(self, param) :
        self.stab_alpha1 = trans(param[0])
        self.vol_alpha1 = trans(param[1])
        self.stab_alpha2 = trans(param[2])
        self.vol_alpha2 = trans(param[3])        
        self.beta = clip_exp(param[4])
    
    def FIT(self, df):

        V_L, V_R, neg_lok = 0, 0, 0

        for t in range(len(df['actions'])):
            P_L = 1 / (1 + np.exp(self.beta * (V_R - V_L)))
            ##----------------------------------##
            ##   Vt = Vt-1 + α * PE(t-1)        ##
            ##   PE(t-1) = Rt-1  -  Vt-1        ##
            ##----------------------------------##
            if t == 0 or t==(len(df['actions'])) // 2:
                P = 0.5  
            else:
                if df.at[t, 'actions'] == 1:
                    PE_L = df.at[t - 1, 'rewards'] - V_L
                    if t <= (len(df['actions'])) // 4:
                        V_L = V_L + self.stab_alpha1 * PE_L
                    elif t >= (len(df['actions'])) // 4 and t <= (len(df['actions'])) // 2:
                        V_L = V_L + self.vol_alpha1 * PE_L
                    elif t <= (len(df['actions'])) // (4/3) and t >= (len(df['actions'])) // 2:
                        V_L = V_L + self.stab_alpha2 * PE_L
                    else:
                        V_L = V_L + self.vol_alpha2 * PE_L
                        
                    P = P_L
                else:
                    PE_R = df.at[t - 1, 'rewards'] - V_R
                    if t <= (len(df['actions'])) // 4:
                        V_R = V_R + self.stab_alpha1 * PE_R
                    elif t >= (len(df['actions'])) // 4 and t <= (len(df['actions'])) // 2:
                        V_R = V_R + self.vol_alpha1 * PE_R
                    elif t <= (len(df['actions'])) // (4/3) and t >= (len(df['actions'])) // 2:
                        V_R = V_R + self.stab_alpha2 * PE_R
                    else:
                        V_R = V_R + self.vol_alpha2 * PE_R    

                    P = 1 - P_L
            
            neg_lok += -np.log(P + eps)
            
        return neg_lok

class Counter:
    p_name = ['alpha', 'beta']
    bnds = [(0, 1), (-10, 10)]
    pbnds = [(0, 0.5), (-5, 5)]
    prior = [norm(0,1.55), norm(0,10)]
    
    def __init__(self, param) :
        self.alpha = trans(param[0])
        self.beta = clip_exp(param[1])
    
    def FIT(self, df):

        V_L, V_R, neg_lok = 0, 0, 0

        for t in range(len(df['actions'])):
            P_L = sigmoid(self.beta, V_L, V_R)

            if t ==(len(df['actions'])) // 2:
                V_L, V_R = 0, 0

            if t > 0:
                if df.at[t, 'actions'] == 1:
                    PE_L = (df.at[t - 1, 'rewards'] - V_L) if (df.at[t - 1, 'rewards'] == 1) else (-df.at[t - 1, 'rewards'] - V_L)
                    PE_R = (-df.at[t - 1, 'rewards'] - V_R) if (df.at[t - 1, 'rewards'] == 1) else (df.at[t - 1, 'rewards'] - V_R)

                else:
                    PE_L = (-df.at[t - 1, 'rewards'] - V_L) if (df.at[t - 1, 'rewards'] == 1) else (df.at[t - 1, 'rewards'] - V_L)
                    PE_R = (df.at[t - 1, 'rewards'] - V_R) if (df.at[t - 1, 'rewards'] == 1) else (-df.at[t - 1, 'rewards'] - V_R)                
                V_L += self.alpha * PE_L
                V_R += self.alpha * PE_R
   
            P = P_L if t == 0 or df.at[t, 'actions'] == 1 else 1 - P_L
            neg_lok += -np.log(P + eps)
        
        return neg_lok

class PH:
    p_name = ['alpha', 'beta','k','eta']
    bnds = [(0, 1), (-10, 10), (0, 1), (0, 1)]
    pbnds = [(0, 0.5), (-5, 5), (0, 0.5), (0, 0.5)]
    prior = [norm(0,1.55), norm(0,10), norm(0,1.55), norm(0,1.55)]
    
    def __init__(self, param) :
        self.alpha = trans(param[0])
        self.beta = clip_exp(param[1])
        self.k = trans(param[2])
        self.eta = trans(param[3])   
    
    def FIT(self, df):

        V_L, V_R, neg_lok = 0, 0, 0

        for t in range(len(df['actions'])):
            P_L = sigmoid(self.beta, V_L, V_R)

            if t ==(len(df['actions'])) // 2:                
                V_L, V_R = 0, 0   

            if t > 0:
                if df.at[t, 'actions'] == 1:
                    PE_L = df.at[t - 1, 'rewards'] - V_L
                    V_L += self.k * self.alpha * PE_L
                    self.alpha = self.eta * np.abs(PE_L) + (1 - self.eta) * self.alpha
                else:
                    PE_R = df.at[t - 1, 'rewards'] - V_R
                    V_R += self.k * self.alpha * PE_R
                    self.alpha = self.eta * np.abs(PE_R) + (1 - self.eta) * self.alpha
                
                self.alpha = np.clip(self.alpha, a_min=-5, a_max=5)
                if self.alpha in [-np.inf, np.inf, np.nan]:
                    pass   
                    
            P = P_L if t == 0 or df.at[t, 'actions'] == 1 else 1 - P_L
            neg_lok += -np.log(P + eps)
        
        return neg_lok

class HMM2:
    p_name = ['gamma', 'c', 'd']
    bnds = [(0, 1), (0.5, 1), (0.5, 1)]
    pbnds = [(0, 0.5), (0.5, 0.75), (0.5, 0.75)]
    prior = [norm(0,1.55), norm(0,1.55), norm(0,1.55)]
    
    def __init__(self, param) :
        self.gamma = trans(param[0])
        self.c = trans(param[1])
        self.d = trans(param[2])   
    
    def FIT(self, df):

        Ps = np.full(2, 0.5)
        neg_lok = 0
        for t in range(len(df['actions'])):
            if t ==(len(df['actions'])) // 2:
                Ps = np.full(2, 0.5)
            if t > 0:
                Ps[0] = (1 - self.gamma) * Ps[0] + self.gamma * Ps[1]
                Ps[1] = 1 - Ps[0]

            P = Ps[0] if df.at[t, 'actions'] == 1 else Ps[1]
            
            if df.at[t, 'rewards'] == 1:
                P_O_S1 = (self.c if df.at[t, 'actions'] == 1 else (1 - self.c))
                P_O_S2 = (self.c if df.at[t, 'actions'] != 1 else (1 - self.c))            
            else:
                P_O_S1 = ((1 - self.d) if df.at[t, 'actions'] == 1 else self.d)
                P_O_S2 = ((1 - self.d) if df.at[t, 'actions'] != 1 else self.d)
                    
            new_Ps0 = (P_O_S1 * Ps[0]) / (P_O_S1 * Ps[0] + P_O_S2 * Ps[1] + eps)
            Ps[0] = new_Ps0
            Ps[1] = 1 - new_Ps0
            
            neg_lok += -np.log(P + eps)
        
        return neg_lok

class HMM2_fixcd:
    p_name = ['gamma', 'c', 'd']
    bnds = [(0, 1), (1, 1), (1, 1)]
    pbnds = [(0, 0.5), (0.5, 0.75), (0.5, 0.75)]
    prior = [norm(0,1.55), norm(0,1.55), norm(0,1.55)]
    
    def __init__(self, param) :
        self.gamma = trans(param[0])
        self.c = 1
        self.d = 1   
    
    def FIT(self, df):

        Ps = np.full(2, 0.5)
        neg_lok = 0
        for t in range(len(df['actions'])):
            if t ==(len(df['actions'])) // 2:
                Ps = np.full(2, 0.5)
            if t > 0:
                Ps[0] = (1 - self.gamma) * Ps[0] + self.gamma * Ps[1]
                Ps[1] = 1 - Ps[0]

            P = Ps[0] if df.at[t, 'actions'] == 1 else Ps[1]
            
            if df.at[t, 'rewards'] == 1:
                P_O_S1 = (self.c if df.at[t, 'actions'] == 1 else (1 - self.c))
                P_O_S2 = (self.c if df.at[t, 'actions'] != 1 else (1 - self.c))            
            else:
                P_O_S1 = ((1 - self.d) if df.at[t, 'actions'] == 1 else self.d)
                P_O_S2 = ((1 - self.d) if df.at[t, 'actions'] != 1 else self.d)
                    
            new_Ps0 = (P_O_S1 * Ps[0]) / (P_O_S1 * Ps[0] + P_O_S2 * Ps[1] + eps)
            Ps[0] = new_Ps0
            Ps[1] = 1 - new_Ps0
            
            neg_lok += -np.log(P + eps)
        
        return neg_lok

class HMM2_fixgamma:
    p_name = ['gamma', 'c', 'd']
    bnds = [(0, 1), (1, 1), (1, 1)]
    pbnds = [(0, 0.5), (0.5, 0.75), (0.5, 0.75)]
    prior = [norm(0,1.55), norm(0,1.55), norm(0,1.55)]
    
    def __init__(self, param) :
        self.gamma = .3
        self.c = trans(param[1])
        self.d = trans(param[2])   
    
    def FIT(self, df):

        Ps = np.full(2, 0.5)
        neg_lok = 0
        for t in range(len(df['actions'])):
            if t ==(len(df['actions'])) // 2:
                Ps = np.full(2, 0.5)
            if t > 0:
                Ps[0] = (1 - self.gamma) * Ps[0] + self.gamma * Ps[1]
                Ps[1] = 1 - Ps[0]

            P = Ps[0] if df.at[t, 'actions'] == 1 else Ps[1]
            
            if df.at[t, 'rewards'] == 1:
                P_O_S1 = (self.c if df.at[t, 'actions'] == 1 else (1 - self.c))
                P_O_S2 = (self.c if df.at[t, 'actions'] != 1 else (1 - self.c))            
            else:
                P_O_S1 = ((1 - self.d) if df.at[t, 'actions'] == 1 else self.d)
                P_O_S2 = ((1 - self.d) if df.at[t, 'actions'] != 1 else self.d)
                    
            new_Ps0 = (P_O_S1 * Ps[0]) / (P_O_S1 * Ps[0] + P_O_S2 * Ps[1] + eps)
            Ps[0] = new_Ps0
            Ps[1] = 1 - new_Ps0
            
            neg_lok += -np.log(P + eps)
        
        return neg_lok

class HMM2_fixgamma_c:
    p_name = ['gamma', 'c', 'd']
    bnds = [(0, 1), (1, 1), (1, 1)]
    pbnds = [(0, 0.5), (0.5, 0.75), (0.5, 0.75)]
    prior = [norm(0,1.55), norm(0,1.55), norm(0,1.55)]
    
    def __init__(self, param) :
        self.gamma = .3
        self.c = 1
        self.d = trans(param[2])   
    
    def FIT(self, df):

        Ps = np.full(2, 0.5)
        neg_lok = 0
        for t in range(len(df['actions'])):
            if t ==(len(df['actions'])) // 2:
                Ps = np.full(2, 0.5)
            if t > 0:
                Ps[0] = (1 - self.gamma) * Ps[0] + self.gamma * Ps[1]
                Ps[1] = 1 - Ps[0]

            P = Ps[0] if df.at[t, 'actions'] == 1 else Ps[1]
            
            if df.at[t, 'rewards'] == 1:
                P_O_S1 = (self.c if df.at[t, 'actions'] == 1 else (1 - self.c))
                P_O_S2 = (self.c if df.at[t, 'actions'] != 1 else (1 - self.c))            
            else:
                P_O_S1 = ((1 - self.d) if df.at[t, 'actions'] == 1 else self.d)
                P_O_S2 = ((1 - self.d) if df.at[t, 'actions'] != 1 else self.d)
                    
            new_Ps0 = (P_O_S1 * Ps[0]) / (P_O_S1 * Ps[0] + P_O_S2 * Ps[1] + eps)
            Ps[0] = new_Ps0
            Ps[1] = 1 - new_Ps0
            
            neg_lok += -np.log(P + eps)
        
        return neg_lok

class HMM1:
    p_name = ['gamma', 'c']
    bnds = [(0, 1), (0.5, 1)]
    pbnds = [(0, 0.5), (0.5, 0.75)]
    prior = [norm(0,1.55), norm(0,1.55)]
    
    def __init__(self, param) :
        self.gamma = trans(param[0])
        self.c = trans(param[1])
    
    def FIT(self, df):
        '''
        要求输入的data格式为dataframe
        列名分别为actions、rewards。 
        每个被试一个dataframe。(字典)
        '''
        Ps = np.full(2, 0.5)
        neg_lok = 0
        for t in range(len(df['actions'])):
            if t == (len(df['actions'])) // 2:
                Ps = np.full(2, 0.5)
            if t > 0:
                Ps[0] = (1 - self.gamma) * Ps[0] + self.gamma * Ps[1]
                Ps[1] = 1 - Ps[0]

            P = Ps[0] if df.at[t, 'actions'] == 1 else Ps[1]
            
            if df.at[t, 'rewards'] == 1:
                P_O_S1 = (self.c if df.at[t, 'actions'] == 1 else (1 - self.c))
                P_O_S2 = (self.c if df.at[t, 'actions'] != 1 else (1 - self.c))            
            else:
                P_O_S1 = ((1 - self.c) if df.at[t, 'actions'] == 1 else self.c)
                P_O_S2 =((1 - self.c) if df.at[t, 'actions'] != 1 else self.c)
                    
            new_Ps0 = (P_O_S1 * Ps[0]) / (P_O_S1 * Ps[0] + P_O_S2 * Ps[1] + eps)
            Ps[0] = new_Ps0
            Ps[1] = 1 - new_Ps0
            
            neg_lok += -np.log(P + eps)
        
        return neg_lok

class HMM0:
    p_name = ['gamma']
    bnds = [(0, 1)]
    pbnds = [(0, 0.5)]
    prior = [norm(0,1.55)]
    
    def __init__(self, param) :
        self.gamma = trans(param[0])
    
    def FIT(self, df):
        '''
        要求输入的data格式为dataframe
        列名分别为actions、rewards。 
        每个被试一个dataframe。(字典)
        '''
        Ps = np.full(2, 0.5)
        neg_lok = 0
        for t in range(len(df['actions'])):
            if t == (len(df['actions'])) // 2:              
                Ps = np.full(2, 0.5)
            if t > 0:
                Ps[0] = (1 - self.gamma) * Ps[0] + self.gamma * Ps[1]
                Ps[1] = 1 - Ps[0]

            P = Ps[0] if df.at[t, 'actions'] == 1 else Ps[1]
            
            if df.at[t, 'actions'] == 1:
                P_O_S1 = 1 if df.at[t, 'rewards'] == 1 else 0
                P_O_S2 = 0 if df.at[t, 'rewards'] == 1 else 1
                
            else:
                P_O_S1 = 0 if df.at[t, 'rewards'] == 1 else 1
                P_O_S2 = 1  if df.at[t, 'rewards'] == 1 else 0

            new_Ps0 = (P_O_S1 * Ps[0]) / (P_O_S1 * Ps[0] + P_O_S2 * Ps[1] + eps)
            Ps[0] = new_Ps0
            Ps[1] = 1 - new_Ps0
            

            neg_lok += -np.log(P + eps) 
                                    
        return neg_lok

class HGF:
    p_name = ['kappa', 'omega2', 'omega3']
    bnds = [(0, 1), (-20, 20), (-20, 20)]
    pbnds = [(0, 0.5), (-5, 5), (-5, 5)]
    prior = [norm(0,1.55), norm(0, 10), norm(0, 10)]
    
    def __init__(self, param) :
        self.kappa = clip_exp(param[0])
        self.omega2 = param[1]
        self.omega3 = param[2] 
    
    def FIT(self, df):
        mu2_t2, da2_t1, P_action, mle_sum= 0, 0, 0, 0
        mu3_t2, pi3_t2 = 1, 1 
        pi2_t2 = 10 

        for t in range(len(df['resp'])):

            if t == (len(df['resp'])) // 2:
                mu2_t2, da2_t1, P_action= 0, 0, 0
                mu3_t2, pi3_t2 = 1, 1 
                pi2_t2 = 10            
            
            mu_hat2_t1 = mu2_t2
            mu_hat1_t1 = 1 / (1 + np.exp(-mu_hat2_t1))
            pi_hat1_t1 = 1 / (mu_hat1_t1 * (1 - mu_hat1_t1) + eps)
            
            mu1_t1_scalar = df.at[t, 'u_t']
            da1_t1 = mu1_t1_scalar - mu_hat1_t1
            
            v2_t1 = np.exp(self.kappa * mu3_t2 + self.omega2)
            pi_hat2_t1 = 1 / ((1 / pi2_t2) + v2_t1)
            pi2_t1 = pi_hat2_t1 + 1 / pi_hat1_t1
            mu2_t1 = mu_hat2_t1 + (1 / pi2_t1) * da1_t1
            da2_t1 = (1 / pi2_t1 + ((mu2_t1 - mu_hat2_t1) ** 2)) * pi_hat2_t1 - 1
            
            mu_hat3_t1 = mu3_t2
            pi_hat3_t1 = 1 / ((1 / pi3_t2) + np.exp(self.omega3))   
            w2_t1 = v2_t1 * pi_hat2_t1
            
            pi3_t1 = pi_hat3_t1 + 0.5 * (self.kappa ** 2) * w2_t1 * ((w2_t1 + (2 * w2_t1 - 1) * da2_t1))
            mu3_t1 = mu_hat3_t1 + 0.5 * (pi3_t1 ** -1) * self.kappa * w2_t1 * da2_t1
            
            mu2_t2 = mu2_t1
            mu3_t2 = mu3_t1
            
            pi2_t2 = pi2_t1
            pi3_t2 = pi3_t1
            if df.at[t, 'resp']== 1:
                P_action = mu_hat1_t1
            else:
                P_action = 1 - mu_hat1_t1
            mle_sum += -np.log(P_action)

        return mle_sum
    #x0 = [kappa_dist.rvs(), omega2_dist.rvs(),omega3_dist.rvs()]

class transRW:
    p_name = ['alpha', 'beta', 'gamma']
    bnds = [(0, 1), (-10, 10), (0, 1)]
    pbnds = [(0, 0.5), (-5, 5), (0, 0.5)]
    prior = [norm(0,1.55), norm(0,10), norm(0,1.55)]
    
    def __init__(self, param) :
        self.alpha = trans(param[0])
        self.beta = clip_exp(param[1])
        self.gamma = trans(param[2])

    def FIT(self, df):

        V_L, V_R, neg_lok = 0, 0, 0

        for t in range(len(df['actions'])): 
            if t ==(len(df['actions']))//2:
                V_L, V_R  = 0, 0
                
            P_action = sigmoid(self.beta,V_L,V_R)
            if df.at[t, 'actions'] == 1:
                PE_A = df.at[t, 'rewards'] - (1 - self.gamma) * V_L
                V_L = (1 - self.gamma) * V_L + self.alpha * PE_A
                V_R = self.gamma * V_R 
                P = P_action            
            else:
                PE_B = df.at[t, 'rewards'] - (1 - self.gamma) * V_R
                V_R = (1 - self.gamma) * V_R + self.alpha * PE_B
                V_L = self.gamma * V_L 
                P = 1 - P_action                   
            neg_lok += -np.log(P + eps) 
        return neg_lok

class BL2:
    '''
    Tim Behrence et al, 2007, Nat Neuro.

    Model of Bayesian learner which include prob learner and volatility learner

    https://www.nature.com/articles/nn1954#Sec14

    '''
    p_name = ['Beta','Gamma']
    bnds = [(-10, 10),(-20, 20)]
    pbnds = [(-5, 5), (-5, 5)]
    prior = [norm(0,10), norm(0,10)]
    def __init__(self, param) :
        '''
        initialize infer process
        delta_t_ijk = sum_i sum_j PV1VK * PR1VR * delta_t-1_ijk * PK
        '''
        self.Beta = clip_exp(param[0])
        self.Gamma = (param[1])
                
        self.discretize()
        self.__init__PV1VK()
        self.__init__PR1VR()
        self.__init__delta_ijk()
        self.__init__PK()

    def discretize(self):
        '''
        set bounds and discreticize
        '''
        self.nsplit = 50
        self.rs = np.linspace(.01, .99, self.nsplit)
        self.vs = np.linspace(-11, -2, self.nsplit)
        self.ks = np.linspace(-2, 2, self.nsplit)    

    def __init__PV1VK(self):
        '''
        PV1VK =  P(Vt|Vt-1=i, K) = N(i, exp(k))
        '''
        PV1VK = np.zeros([self.nsplit, self.nsplit, self.nsplit]) 
        for k, ki in enumerate(self.ks):
            for i, vi in enumerate(self.vs):
                PV1VK[:, i, k] = norm.pdf(self.vs, loc=vi, scale = np.exp(ki))
        
        PV1VK = PV1VK / PV1VK.sum(axis=0, keepdims=True)
        self.PV1VK = PV1VK

    def __init__PR1VR(self):
        '''
        PR1VR = P(Rt|Vt-1=i, Rt-1=j) = beta(r,v) = beta(j, i)
        '''
        PR1VR = np.zeros([self.nsplit, self.nsplit, self.nsplit])
        for j, ri in enumerate(self.rs):
            for i, vi in enumerate(self.vs):
                PR1VR[:, i, j] = rbeta(ri, vi).pdf(self.rs)

        PR1VR = PR1VR / PR1VR.sum(axis=0, keepdims=True)
        self.PR1VR = PR1VR

    def __init__delta_ijk(self):
        '''
        delta_ijk * PK
        '''
        delta_ijk = np.ones([self.nsplit, self.nsplit, self.nsplit]) 
        self.delta_ijk = delta_ijk/delta_ijk.sum()

    def __init__PK(self):
        '''
        p(K) = Uniform
        Initialize as a vector of length n_split
        '''
        self.PK = np.ones_like([self.ks]) / self.nsplit 

    def FIT(self, df):
        neg_lok = 0
        for t in range(len(df['actions'])):  
            if t == len(df['actions']) // 2:
                self.__init__delta_ijk()
            action = df.at[t, 'actions']
            ##  p(yt|rt)，likelihood
            P_yr = bernoulli.pmf(action, self.rs)

            ##  generate process: v-r-y
            ##  part1 p(Vt | Vt-1, K) * delta_t-1, 积v→i
            ##  delta: (i,j,k)
            ## (v, i, k) @ (i, j, k)
            delta1 = np.einsum('vi..., ij...->vj...', self.PV1VK, self.delta_ijk)
            
            ##  part2 p(Rt | Rt-1, Vt) * delta_t, 积r→j
            ## （r, i, j）@ (v, j, k) 
            delta2 = np.einsum('r...j, ...jk->...rk', self.PR1VR, delta1)
            
            ## self.delta_ijk: (i,j,k), delta multiply by lj(r)
            ## observation only directly correlates with y，not Vt or K 。
            delta = P_yr[ np.newaxis, :, np.newaxis] * delta2
            
            ## update self.delta_ijk
            self.delta_ijk = delta/delta.sum()
        
            ## calculate V 和 R
            self.P_V = (self.delta_ijk.sum(axis =(1,2))*self.vs).sum()
            self.P_R = (self.delta_ijk.sum(axis=(0,2))*self.rs).sum()

            ## g_t = F(rg)_t * f_t
            ## f_t = reward size = 1
            ## F(rg)_t = max[min[(gamma*(r-0.5)+0.5), 1], 0]
            F_rg1 = np.max([np.min([(self.Gamma*(self.P_R-0.5)+ 0.5), 1]), 0])
            F_rg2 = np.max([np.min([(self.Gamma*(1-self.P_R -0.5)+ 0.5), 1]), 0])

            # F_rg1 = self.P_R
            # F_rg2 = 1-self.P_R

            ## likelihood accorfing to sigmoid func
            P_1 = sigmoid(self.Beta, F_rg1, F_rg2)
            P = P_1 if action ==1 else (1-P_1)
            neg_lok += -np.log(P + eps)

        return neg_lok

class BL1:
    '''
    This version is similar with BayesLearner except that it only has 1 param(beta, inverse temp)

    '''
    p_name = ['Beta']
    bnds = [(-10, 10)]
    pbnds = [(-5, 5)]
    prior = [norm(0,10)]
    def __init__(self, param) :
        '''
        initialize infer process
        delta_t_ijk = sum_i sum_j PV1VK * PR1VR * delta_t-1_ijk * PK
        '''
        self.Beta = clip_exp(param[0])
                
        self.discretize()
        self.__init__PV1VK()
        self.__init__PR1VR()
        self.__init__delta_ijk()
        self.__init__PK()

    def discretize(self):
        '''
        set bounds and discreticize
        '''
        self.nsplit = 50
        self.rs = np.linspace(.01, .99, self.nsplit)
        self.vs = np.linspace(-11, -2, self.nsplit)
        self.ks = np.linspace(-2, 2, self.nsplit)    

    def __init__PV1VK(self):
        '''
        PV1VK =  P(Vt|Vt-1=i, K) = N(i, exp(k))
        '''
        PV1VK = np.zeros([self.nsplit, self.nsplit, self.nsplit]) 
        for k, ki in enumerate(self.ks):
            for i, vi in enumerate(self.vs):
                PV1VK[:, i, k] = norm.pdf(self.vs, loc=vi, scale = np.exp(ki))
        
        PV1VK = PV1VK / PV1VK.sum(axis=0, keepdims=True)
        self.PV1VK = PV1VK

    def __init__PR1VR(self):
        '''
        PR1VR = P(Rt|Vt-1=i, Rt-1=j) = beta(r,v) = beta(j, i)
        '''
        PR1VR = np.zeros([self.nsplit, self.nsplit, self.nsplit])
        for j, ri in enumerate(self.rs):
            for i, vi in enumerate(self.vs):
                PR1VR[:, i, j] = rbeta(ri, vi).pdf(self.rs)

        PR1VR = PR1VR / PR1VR.sum(axis=0, keepdims=True)
        self.PR1VR = PR1VR

    def __init__delta_ijk(self):
        '''
        delta_ijk * PK
        '''
        delta_ijk = np.ones([self.nsplit, self.nsplit, self.nsplit]) 
        self.delta_ijk = delta_ijk/delta_ijk.sum()

    def __init__PK(self):
        '''
        p(K) = Uniform
        Initialize as a vector of length n_split
        '''
        self.PK = np.ones_like([self.ks]) / self.nsplit 

    def FIT(self, df):
        neg_lok = 0
        for t in range(len(df['actions'])): 
            if t == len(df['actions']) // 2:
                self.__init__delta_ijk()

            action = df.at[t, 'actions']
            ##  p(yt|rt)，likelihood
            P_yr = bernoulli.pmf(action, self.rs)

            delta1 = np.einsum('vi..., ij...->vj...', self.PV1VK, self.delta_ijk)
            delta2 = np.einsum('r...j, ...jk->...rk', self.PR1VR, delta1)
            delta = P_yr[ np.newaxis, :, np.newaxis] * delta2
            
            ## update self.delta_ijk
            self.delta_ijk = delta/delta.sum()

            self.P_V = (self.delta_ijk.sum(axis =(1,2))*self.vs).sum()
            self.P_R = (self.delta_ijk.sum(axis=(0,2))*self.rs).sum()

            F_rg1 = self.P_R
            F_rg2 = 1-self.P_R

            ## likelihood accorfing to sigmoid func
            P_1 = sigmoid(self.Beta, F_rg1, F_rg2)
            P = P_1 if action ==1 else (1-P_1)
            neg_lok += -np.log(P + eps)

        return neg_lok

class Args:
    def __init__(self, n_fit,n_sim,n_cores):
        self.n_fit = n_fit
        self.n_sim = n_sim
        self.n_cores = n_cores

