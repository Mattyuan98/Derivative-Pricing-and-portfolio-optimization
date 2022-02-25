import numpy as np
import pandas as pd
import random
from scipy.stats import norm
import matplotlib.pyplot as plt


def annu_rets(r,T):
    """
    convert the non-annual returns to annualized returns
    r: non-annual returns series
    T: periods per year for the returns, i.e. T = 12 for onverting monthly data to annual
    """
    compounded_growth = (1+r).prod()
    t = r.shape[0] #number of periods, i.e. t=24 for 2-year monthly record
    return compounded_growth**(T/t)-1

def annu_vol(r,T):
    """
    annulaized the volatility of returns
    r : returns series
    T: periods per year for converting to annual return, i.e. T = 12 for onverting monthly data to annual 
    """
    return r.std() * np.sqrt(T)

def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

def drawdown(returns_series: pd.Series):
    """
    Takes a time series returns
    Computes and returns a data frame that comtains:
    wealth index
    previous peaks 
    percent drawdowns
    """
    wealth_index = 1000*(1+returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "wealth": wealth_index,
        "peaks" : previous_peaks,
        "drawdowns" : drawdowns
    })
def sharp_ratio(r, riskfree_rate, T):
    """
    Computes the annualized sharp ratio of a set of returns
    """
    #convert the annual risk free rate to per period
    rf_per_period= (1+riskfree_rate)**(1/T)-1
    excess_ret= r- rf_per_period
    ann_ex_ret = annu_rets(excess_ret, T)
    ann_vol = annu_vol(r,T)
    return ann_ex_ret/ann_vol

def skewness(r):
    demeaned_r = r - r.mean()
    #use the population standard deviation, set dof = 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r = r - r.mean()
    #use the population standard deviation, set dof = 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    
    return exp/sigma_r**4


def var_gaussian(r,level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a Series or DataFrame
    If "modified" is true, then the modified VaR is returned
    Using the Cornishi-Fisher modification
    """
    # Compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        #modify the z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z+
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 - 
                (2*z**3 - 5*z)*(s**2)/36
            )
        
    return -(r.mean() + z*r.std(ddof=0))

def var_historic(r,level=5):
    """
    Var Historic
    """
    if isinstance(r,pd.DataFrame): #if is the dataframe, returns True, otherwide false
        return r.aggregate(var_historic, level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected r to be Series or Dataframe")
        

def cvar_historic(r,level=5):
    """
    Computes the conditional VaR of Series or DataFrame
    """
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or Dataframe")

def Summary_Stats(r,riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annu_rets,T=250)
    ann_vol = r.aggregate(annu_vol, T=250)
    ann_sr = r.aggregate(sharp_ratio, riskfree_rate=riskfree_rate, T=250)
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified = True)
    hist_cvar5 = r.aggregate(cvar_historic)
    dd = r.aggregate(lambda r: drawdown(r).drawdowns.min())
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Var(5%)": cf_var5,
        "CVaR(5%)": hist_cvar5,
        "Sharp Ratio": ann_sr,
        "Max Drawdown":dd
    })

#Basic structure of Differential Evolution Algorithm
#INPUTS: Objective_fun, LB, HB, Population size, F(mutate factor), CR(crossover  rate), Iter
#1- Initialize a random population P
#2- evaluate fitness (f) of P
#3- Cycle through each individual in the population
#    3-a) Perform Mutation phase
#      b) Perform Crossover phase
#      c) Perform Selection phase
#4- if stopping criterition has been met:
#       Exit and return best solution
#    else:
#       iteration = iteration + 1
#      go back to step #3

def ensure_bounds (vec,bounds):
    """
    ensure the donor vectors are within the bounds
    if the Ui > HighBound, then Ui = HB
    If the Ui < Lowbound, then Ui = LB
    """
    vec_new = []
    #cycle through each variable un vector
    for i in range (len(vec)):
        #variable exceeds the minimum boundary,the donor vector is replaced with the minimum bound
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])
        #variable exceeds the maximum boundary,the donor vector is replaced with the maxmum bound
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])
        #the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i]) 
    return vec_new


def main(func, bounds, popsize, mutate, CR, maxiter):
    """
    Arguments of the main function:
    bounds:    given a list of tuples representing the search space bounds for each input variables Xn such that: bounds = [(X1min,X1max),(X2min,X2max)...(Xnmin,Xnmax)] (length of bounds is the dimension)
    popsize:   size of population (must be greater than or equal to 4,because the mutation phase needs at least 3 variables other than itself to generate a new variable)
    mutate:    mutation factor F for which apply V = X1 + F(X2 - X3) and F ranges from [0,2] default as 0.5
    maxiter:   the number of generations we want to run the algo (max number of generations)
    CR:        crossover rate varies between [0,1] and determines if crossover occurs. First, generate  a random value between [0,1].If this random value is less than the Crossover Rate, crossover occurs and swap out the current variable in our target vector with the corresponding variable in  the donor vector. If the random value is greater than recombination rate, crossover does not happen and the variable in the target vector remains. The new offspring is trial vector.
    """
# ------------------------Initialize a random population (step 1)-------------------------#
    population = [] #Empty array
    for i in range(0,popsize): 
        indv = []
        
        for j in range (len(bounds)):
            indv.append(random.uniform(bounds[j][0],bounds[j][1]))  
        population.append(indv) #generate a vector with random numbers within the bounds
#cycle through each population (step 2)
    
    gen_best_list = [] #store the best fitness value
    x_list = []
    for i in range(1, maxiter+1):
        #自适应变异算子操作
        lambda_f = np.exp(1 - maxiter/(maxiter + 1 - i ))
        F = mutate * (2 **lambda_f)
        #note that the min and max of (lambda_f) is (0,1)
        #therefore the F ranges from 2*mutate to 1*mutate
        x_list.append(i)
        
        # print ('GENERATION:',i)
        gen_scores =[]
        for j in range(0,popsize):
            #---------------------- Mutation Phase ------------------step3a---------------------#
            #select three random vector index position [0,popsize), not include current vector(j)
            candidates = list(range(0,popsize)) 
            candidates.remove(j)
            random_index = random.sample(candidates,3) 
            #randomly select 3 index numbers from the candidates
            
            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]  #target individual
            
            #Substract X3 from X2 and create a new vector (X_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2,x_3)]
            
            #multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + F * x_diff_i for x_1_i, x_diff_i in zip(x_1,x_diff)]
            #generate a donor vector V
            v_donor = ensure_bounds(v_donor, bounds)
            #Bounds the donor vector if Vi < LB, Vi = LB, if Vi > UB, Vi = UB
            #-------------------Crossover phase----------------# step 3b--------------------#
            v_trial = []
            #cycle through each variable in our target vector
            ### New Cross Over factor
            CrossOver = CR * (1+ random.random())
            #Note that the CR is defaulted as 0.5 and the new crossover rate has a mean of 0.75

            for k in range(len(x_t)):
                rand_rate = random.random()
                #generate a random number btw 0 and 1
                
                rand_delta = random.randint(0,len(x_t)-1) 
                #generate a random parameter delta
                
                #crossover did not occur when rand_rate > crossover rate
                if rand_rate > CrossOver or k != rand_delta:
                    v_trial.append(x_t[k])
                    
                #crossover occurs when rand_rate <= CR and k==delta
                else:
                    v_trial.append(v_donor[k])
                    
            #----------------------Selection phase------------------step3c------------------#
            #for minimization problem, if FUi < Fi then Xi=Ui,Fi = FUi 即选择fitness value里较小的
            
            score_trial = func(v_trial)   #FUi
            score_target = func(x_t)      #Fi
            #greedy selection to update the fitness value of p
            
            if score_trial < score_target:
                population[j] = v_trial 
                gen_scores.append(score_trial)
                #print ('   >',score_trial, v_trial)
            else:
                #print ('   >',score_target, x_t)
                gen_scores.append(score_target)
                
            gen_best = min(gen_scores)                                  # fitness of best individual
            gen_sol = population[gen_scores.index(min(gen_scores))]     # solution of best individual
            #print ('        > GENERATION BEST:',gen_best)
            #print ('          > BEST SOLUTION:',gen_sol,'\n')
        gen_best_list.append(gen_best) 
        #print(gen_best_list)     
        
    print ('差分进化算法最优值 = ', str(gen_best))
    print ('差分进化算法可行解 = ', str(gen_sol))
    
    plt.plot(x_list,gen_best_list,color = "blue",label = "DE")
    plt.xlabel("Iteration Number")
    plt.ylabel("Fitness Value")
    plt.legend()
    plt.show()
    
    return gen_sol


