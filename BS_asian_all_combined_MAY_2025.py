#/usr/bin/env python
# coding: utf-8

# In[1]:



#import the library pulp as p
# import pulp as p
# from pulp import *
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB, quicksum
from scipy.optimize import minimize

#import the necessary functions
from Interpolation_functions import *

# from Functions_for_BS import *
# from Common_functions import *


# In[2]:


import gurobipy as gp
print(gp.gurobi.version())
model = gp.Model()
print("License ID:", model.getParamInfo("LicenseID"))


# In[3]:


#common parameters for all the experiments/GLOBAL VARIABLES

stock = 1 #initial stock price

K = 1 # target option's strike price

r = 0 #risk-free rate of interest

sigma = 0.2 #volatility of the stock price

T = 1 #target maturity

mu = 0 #return of the stock price

t = 0

timepoints =2 #number of timesteps


T_GRID = [0,0.5,T]#times at which the marginals of the stock price are evaluated

short_maturity = T_GRID[1] # maturity of the options in the hedge portfolio


#short_strike_range = [1,1.1,0.9,1.2,0.8,1.25,1.05,0.85,]#range of short strikes for increasing number of options


# In[4]:


def Asian_option_payoff(stock_price_t1,stock_price_T,strike_price):
    return np.maximum(0.5*(stock_price_t1+stock_price_T)-strike_price,0)


# **GET THE ASIAN OPTION'S VALUE FROM THE MOT PROBLEM**

# In[ ]:


#import the results from the Asian option MOT and compare the asian option values from the MOT and our hedging problem
# import pickle
# with open("result_Asian_MOT.pkl","rb") as f:
#     data_MOT = pickle.load(f)

    
# Target_option_value = data_MOT['Final_value']    
# print(f"Price of Asian option using MOT:{Target_option_value}")



# In[ ]:


#function to return the discretization points, call prices and the marginals for a given value of truncation range and number of discretization points and choice of uniform or nonuniform grid

def get_disc_points_and_marginals(truncation_range=None,n=None, N=None,choice="uniform"):
    """
       Generate the discretization points for the given problem depending on uniform or non-uniform grid
       Parameters: 
          Global variables: Stock, sigma, T,r,
          Local variables: n- (1/n) denotes the spacing between the discretization points
                        
    """
    if choice == "uniform":
        if truncation_range == None or n == None:
            return ("Need to provide n or truncation range")
        #return a uniform grid
        D = np.linspace(0, truncation_range, int(truncation_range*n)+1)
        
        
        #next we find the point where the interpolated line joining the last two strike points takes value zero
        R_0 = find_zero(BS_call,stock,D,T,sigma,r)
        
#         print(f"The interpolated line takes the value 0 at {R_0}")
        
        #print the corresponding call value
#         print(f"Strike:{R_0},Interpolated call value:{interpolated_line(stock,D,T,sigma,r,R_0)}")
        
        
        #extending the truncation range to include the new points till the zero of the interpolated call value
        K1_new = append_numbers(D,D[-1],R_0,np.diff(D)[0])
        
#         print(f"Extended truncation range for time {t1[1]}:{K1_new}")
#         print(f"length of new truncation range :{len(K1_new)}")
#         print(f"previous length:{len(D)}")
        
        K2_new = append_numbers(D,D[-1],R_0,np.diff(D)[0])
#         print(f"Extended truncation range for time {t1[2]}:{K2_new}")
#         print(f"spacing:{K2_new[-2]/(len(K2_new)-2)},{D[-1]/(len(D)-1)}") 
        
        
        #getting the new discretization points
        disc_points_new = disc_points_new_1(K1_new,K2_new)
        print(f"The new discretization points are:{disc_points_new}")
        
        # store the call prices as a 2D array
        call_prices_array_new = np.vstack((call_prices(stock,D,K1_new,T_GRID[1],sigma,r),
                                           call_prices(stock,D,K2_new,T,sigma,r)))
#         print(call_prices_array_new)
        
        #getting the discrete marginal distributions

        marg = marginal_new(call_prices_array_new,disc_points_new,timepoints,T_GRID)
#         print(marginal_new(call_prices_array_new,disc_points_new,timepoints,t1))
        
        
        #returns the grid and the number of grid points
        return disc_points_new,call_prices_array_new,marg
    else:
        if N == None:
            return print("Need to provide N")
        np.random.seed(12)
        grid = np.random.normal(K,sigma,N-2)# creating a grid concentrated around the target strike
        
        grid = np.append([0,truncation_range],grid)# adding 0 and the final truncation point to the list
        
        grid = np.sort(grid)# sorting the grid in ascending order
        
        
         #next we find the point where the interpolated line joining the last two strike points takes value zero
        R_0_T1 = find_zero(BS_call,stock,grid,T_GRID[1],sigma,r)
        print(f"The strike for time {T_GRID[1]} is {R_0_T1} and interpolated call value: {interpolated_line(stock,grid,T_GRID[1],sigma,r,R_0_T1)}")
        
        R_0_T = find_zero(BS_call,stock,grid,T,sigma,r)
        print(f"The strike for time {T} is {R_0_T} and interpolated call value: {interpolated_line(stock,grid,T,sigma,r,R_0_T)}")
        
        R_0 = max(R_0_T1,R_0_T)
        
#         #next we find the point where the interpolated line joining the last two strike points takes value zero
#         R_0 = find_zero(BS_call,stock, grid,T,sigma,r)
        
#         print(f"The interpolated line takes the value 0 at {R_0}")
        
#         #print the corresponding call value
#         print(f"Strike:{R_0},Interpolated call value:{interpolated_line(stock,grid,T,sigma,r,R_0)}")
        
        new_grid = np.append(grid,[R_0])
        
        K1_new = K2_new = new_grid
        
        #getting the new discretization points
        disc_points_new = disc_points_new_1(K1_new,K2_new)
        print(f"The new discretization points are:{disc_points_new}")
        print(f"Number of discretization points are:{len(K1_new)}")
        
                
        # store the call prices as a 2D array
        call_prices_array_new = np.vstack((call_prices(stock,grid,K1_new,T_GRID[1],sigma,r),call_prices(stock,grid,K2_new,T,sigma,r)))
#         print(call_prices_array_new)
        
        #getting the discrete marginal distributions

        marg = marginal_new(call_prices_array_new,disc_points_new,timepoints,T_GRID)
#         print(marginal_new(call_prices_array_new,disc_points_new,timepoints,T_GRID))
        
        #returns the grid and the number of grid points
        return disc_points_new,call_prices_array_new,marg
# print(BS_call(1000,990,0,1,0.05,0))


# In[7]:


def get_target_call(marg, disc_points_new,K):
    """
    returns the price of the target option and the short-maturity options in the hedge portfolio
    Parameters:
       marg: marginal distributions, 
       disc_points_new: the discretization points 
    """
    N = len(disc_points_new[1]) #returns the number of discretization points corresponding to the target maturity
    target_call = sum(marg[1,i] * np.maximum(disc_points_new[1,i]- K,0) for i in range(N))
    return target_call
    


def get_short_call(t1,short_maturity,no_of_options,marg,disc_points_new,short_strikes):
    """
    returns the price of the target option and the short-maturity options in the hedge portfolio
    Parameters:
      t1: list of the time-points given by [0,short_maturity,target_maturity]
      short_maturity: maturity of the options in the hedge portfolio
      no_of_options: numner of options in the hedge portfolio
      marg: marginal distributions
      disc_points_new: discretization points
      short_strikes: strikes of the options in the hedge portfolio
    """
    short_call = np.zeros(no_of_options)
    
    L = t1.index(short_maturity)
    
    N = len(disc_points_new[0])#returns the number of discretization points corresponding to the short maturity
    
    #Value of short maturity options at time 0
    # short_call = \sum_{i}\alpha_i(x_i-short_strike)^+
    for j in range(no_of_options):
        short_call[j] = sum(marg[L-1,i] * np.maximum(disc_points_new[L-1,i]-short_strikes[j],0) \
                            for i in range(N))
        print("Short call value", str(j),"is:",short_call[j])
    return short_call





# **SOLVE THE MAXIMIZATION PROBLEM WITHOUT M_BOUND**

# In[23]:


def solve_max_problem_gurobi_without_M(M_bound,
    K1, K2,
    Weights_min_prev, short_strikes,short_call, target_strike,
    marg, Epsilon
):
    """
    A version of the 'solve_max_problem_gurobi' function that uses row-specific M_i
    instead of one global M_bound.
    """
    # Create a Gurobi model
    model = gp.Model("Upper-Bound-Problem-RowSpecificM")
    model.setParam('OutputFlag', 0)

    N = len(K1)
    no_of_options = len(short_strikes)

    # Decision variables
    X = model.addVars(N, N, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="X")
    P_slack = model.addVars(N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_slack")
    B = model.addVars(N, vtype=GRB.BINARY, name="B")
#     Delta = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="Delta")

    # Objective: Maximize sum of P_slack
    model.setObjective(quicksum(P_slack[i] for i in range(N)), GRB.MAXIMIZE)

    
  

    
     
    for i in range(N):
         # The hedging_error expression
        hedging_error = (quicksum(X[i, j] * Asian_option_payoff(K1[i],K2[j],target_strike) for j in range(N)) \
                        - marg[0][i] * (
                            Weights_min_prev[0]
                            + quicksum(
                                Weights_min_prev[l+1] * max(K1[i] - short_strikes[l], 0)
                                for l in range(no_of_options)
                            )
                           )
                        )
        # -- Absolute value constraints:
        # a) hedging_error <= P_slack[i]
        model.addConstr(hedging_error <= P_slack[i], f"modulus_1_{i}")

        # b) hedging_error >= -P_slack[i]
        model.addConstr(hedging_error >= -P_slack[i], f"modulus_2_{i}")

#         # d) hedging_error *(1 - 2*B[i]) == P_slack[i]
        model.addConstr(hedging_error *(1 - 2*B[i]) == P_slack[i], f"modulus_3_{i}")
        
    
#     # Sign[i] == 1  ->  e_i >= 0  &  P ==  e_i
#         model.addGenConstrIndicator(B[i], True,  hedging_error, GRB.GREATER_EQUAL, 0.0)
#         model.addGenConstrIndicator(B[i], True,  hedging_error - P_slack[i], GRB.EQUAL, 0.0)

#         # Sign[i] == 0  ->  e_i <= 0  &  P == -e_i
#         model.addGenConstrIndicator(B[i], False, hedging_error, GRB.LESS_EQUAL,   0.0)
#         model.addGenConstrIndicator(B[i], False, P_slack[i] + hedging_error , GRB.EQUAL, 0.0)
   
    # Marginal constraints
    for i in range(N):
        model.addConstr(quicksum(X[i, j] for j in range(N)) == marg[0][i], f"time1_{i}")

    for j in range(N):
        model.addConstr(quicksum(X[i, j] for i in range(N)) == marg[1][j], f"time2_{j}")

    # Martingale constraints 
   
    for i in range(N):
        model.addConstr(quicksum(X[i, j]*(K2[j] - K1[i]) for j in range(N)) == 0, f"martingale_{i}")
    
#     model.setParam('NonConvex', 2)

    # Solve the model
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return {
            "call_value": float("inf"),
            "Status": model.status,
            "p_matrix": None,
            "P_slack": None,
            "B": None,
#             "Delta": None
        }
    
 
    # 6) Extract solution
    p_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            p_matrix[i, j] = X[i, j].X

    p_slack_array = np.array([P_slack[i].X for i in range(N)])
    b_array       = np.array([B[i].X          for i in range(N)])
#     delta_array   = np.array([Delta[i].X      for i in range(N)])
    call_value    = model.objVal

    return {
        "call_value": call_value,
        "Status": "Optimal",
        "p_matrix": p_matrix,
        "P_slack": p_slack_array,
        "B": b_array,
#         "Delta": delta_array,
       
    }


# In[24]:


import cvxpy as cp


# In[25]:


def min_max_optimization(
    solve_max_problem,
    M_bound,
    K1,
    K2,
    short_strikes,
    no_of_options,
    K,
    marg,
    short_call,
    Epsilon,
    x0, #initial choice for weights
    seed,
    bounds,
    regularization,
    lmda_reg
):
    """
    Returns the result of the min-max process for a given number of options
    in the hedge portfolio.
    """

    # Keep a log of intermediate values
    log = []

    n_timepoint_1 = len(marg[0])
    n_timepoint_2 = len(marg[1])

    def min_function_on_weights(Weights_with_cash):
        """
        Inner objective: given a set of hedge weights, solve the max problem
        and compute the "error" (sum of absolute differences).
        """

        # Solve the inner max problem
        result = solve_max_problem(
            M_bound,
            K1,
            K2,
            Weights_with_cash,
            short_strikes,
            short_call,
            K,
            marg,
            Epsilon
        )
        Status = result["Status"]
        if Status != "Optimal":
            # If infeasible or not optimal, penalize heavily
            print(f"Warning: Status is {Status}, no solution found.")
            return float("inf")

        # Retrieve the numeric arrays from the solver result dictionary
        p_matrix = result["p_matrix"]      # shape: (n_timepoint_1, n_timepoint_2)
        P_slack  = result["P_slack"]       # shape: (n_timepoint_1,)
        B_array  = result["B"]             # shape: (n_timepoint_1,)
#         Delta    = result["Delta"]         # shape: (n_timepoint_1,)
        call_val = result["call_value"]    # the objective in the solve_max_problem

        # Compute the "error" function f_w_pr:
        #    sum over i of | sum_{j} p_matrix[i,j]*(K2[j] - K)^+
        #                     - alpha_i*(hedge payoff at i) |
        alpha = marg[0]
        f_w_pr = 0.0
        
        
        #compute the optimal cash position 
        cash =  Weights_with_cash[0] 
        Weights_of_options = Weights_with_cash[1:] 
        
        #compute the objective value and hedge value
        hedge_value = 0
        for i in range(n_timepoint_1):
            payoff_target = sum(
                p_matrix[i,j] * Asian_option_payoff(K1[i],K2[j],K) 
                for j in range(n_timepoint_2))
            

            payoff_hedge = alpha[i] * (cash
                + sum(
                    Weights_of_options[l] * max(K1[i] - short_strikes[l], 0)
                    for l in range(no_of_options)
                )
            )
            f_w_pr += abs(payoff_target - payoff_hedge)
            hedge_value += payoff_hedge
        
        
        #compute the target option price
        target_price = 0
        for i in range(n_timepoint_1):
            for j in range(n_timepoint_2):
                target_price +=  p_matrix[i,j] * Asian_option_payoff(K1[i],K2[j],K)
        
       
        
        #if regularization is done through lasso or ridge
        if regularization:
            if lmda_reg == None:
                raise ValueError("lmda_reg must be provided")
            reg_type = regularization.lower()
            if reg_type == "lasso":
                #add a regularization term
                regularisation_term = lmda_reg * sum(abs(w) for w in Weights_with_cash)
                f_w_pr += regularisation_term
            
            elif regularization == "ridge":
            
                  #add a regularization term
                regularisation_term = lmda_reg * sum(w**2 for w in Weights_with_cash)
                f_w_pr += regularisation_term
            else:
                raise ValueError("Regularization must be lasso, ridge or none")
        
        
        # Record everything in our log
        log_entry = {
            "Weights": Weights_with_cash,
            "cash": cash,
            "Objective value (outer)": f_w_pr,
            "p_matrix": p_matrix,
            "P_slack": P_slack,
            "B": B_array,
#             "Delta": Delta,
            "call_value (inner)": call_val,
            "Status": Status,
            "target_price": target_price,
            "hedge_value": hedge_value
        }
        log.append(log_entry)
        print(f"value from gurobi:{call_val}")
        print(f"val solving max LP:{f_w_pr}\n")
       
        return f_w_pr

   
      # pick solver function
    
    result_dual_annealing = minimize(fun=min_function_on_weights,bounds= bounds,
                                     method='Nelder-Mead',
                                     x0 = x0,
                                      options=dict(maxiter=10000*(no_of_options+1),
                                                   maxfev=10000*(no_of_options+1),fatol= 1e-8, xatol= 1e-6,
                                                   disp=True)
                                                   
                                    )

    if result_dual_annealing.success:
        w_opt = result_dual_annealing.x
        final_value = result_dual_annealing.fun
        print(result_dual_annealing.message)
        return w_opt, final_value, log
    else:
        print("Simulated annealing failed")
        return None, float("inf"), log


# In[ ]:


def check_constraints_pulp_solution_no_M(
    M_bound,
    K1, K2,
    Weights,         # [cash, w1, w2, ...]
    short_strikes,
    target_strike,
    marg,
    epsilon,
    solution_dict,
    tolerance=1e-8
):
    """
    Re-checks constraints from solve_max_problem_gurobi_without_M.
    Returns (all_satisfied, violations).
    """

    # ============= 1) Basic Setup =============
    n_timepoint_1 = len(marg[0])  # alpha
    n_timepoint_2 = len(marg[1])  # beta
    no_of_options = len(short_strikes)

    # Extract final solution arrays
    p_matrix = solution_dict["p_matrix"]  # shape (n_timepoint_1, n_timepoint_2)
    P_slack  = solution_dict["P_slack"]   # shape (n_timepoint_1,)
    B_array  = solution_dict["B"]         # shape (n_timepoint_1,)

    # If solver not optimal or p_matrix is None -> skip
    if solution_dict["Status"] != "Optimal" or p_matrix is None:
        return (
            False,
            [f"Solver status {solution_dict['Status']} or p_matrix=None => skip check"]
        )

    violations = []

    # ============= 2) Timepoint 1 Constraints =============
    # sum_j p[i,j] == marg[0][i]
    for i in range(n_timepoint_1):
        lhs = np.sum(p_matrix[i,:])
        rhs = marg[0][i]
        if abs(lhs - rhs) > tolerance:
            msg = f"Row sum constraint violated at i={i}: LHS={lhs}, RHS={rhs}"
            violations.append(msg)

    # ============= 3) Timepoint 2 Constraints =============
    # sum_i p[i,j] == marg[1][j]
    for j in range(n_timepoint_2):
        lhs = np.sum(p_matrix[:, j])
        rhs = marg[1][j]
        if abs(lhs - rhs) > tolerance:
            msg = f"Column sum constraint violated at j={j}: LHS={lhs}, RHS={rhs}"
            violations.append(msg)

    # ============= 4) Martingale Constraints =============
    # sum_j p[i,j]*(K2[j] - K1[i]) = 0  for each i
    for i in range(n_timepoint_1):
        lhs = 0.0
        for j in range(n_timepoint_2):
            lhs += p_matrix[i, j]*(K2[j] - K1[i])
        if abs(lhs) > tolerance:
            msg = f"Martingale constraint violated at i={i}: LHS={lhs}"
            violations.append(msg)

    # ============= 5) Absolute Value (No Big M) Constraints =============
    # For each i, define:
    # hedging_error = sum_j p[i,j]*(K2[j]-target_strike)^+
    #               - alpha[i]*(cash + sum_l W[l+1]*(K1[i]-short_strikes[l])^+)
    # 1) hedging_error <= P_slack[i]
    # 2) hedging_error >= -P_slack[i]
    # 3) hedging_error*(1 - 2*B[i]) == P_slack[i]

    alpha = marg[0]
    cash  = Weights[0]
    w_opts = Weights[1:]  # the short positions
  

    for i in range(n_timepoint_1):
        # payoff_target
        payoff_target = 0.0
        for j in range(n_timepoint_2):
            payoff_target += p_matrix[i,j] * Asian_option_payoff(K1[i], K2[j], target_strike) 

        # payoff_hedge
        payoff_hedge = alpha[i]*(
            cash + sum(
                w_opts[l] * max(K1[i] - short_strikes[l], 0)
                for l in range(no_of_options)
            )
        )

        hedging_error = payoff_target - payoff_hedge

        # 1) hedging_error <= P_slack[i]
        if hedging_error - P_slack[i] > tolerance:
            msg = (f"Absolute-value constraint #1 violated i={i}: "
                   f"hedging_error={hedging_error}, P_slack={P_slack[i]}")
            violations.append(msg)

        # 2) hedging_error >= -P_slack[i]
        if hedging_error + P_slack[i] < -tolerance:
            msg = (f"Absolute-value constraint #2 violated i={i}: "
                   f"hedging_error={hedging_error}, P_slack={P_slack[i]}")
            violations.append(msg)

        # 3) hedging_error*(1-2*B[i]) == P_slack[i]
        lhs_3 = hedging_error*(1 - 2*B_array[i])
        if abs(lhs_3 - P_slack[i]) > tolerance:
            msg = (f"Absolute-value constraint #3 violated i={i}: "
                   f"lhs_3={lhs_3}, P_slack={P_slack[i]}, B={B_array[i]}")
            violations.append(msg)

    # ============= Finalize =============
    all_satisfied = (len(violations) == 0)
    return (all_satisfied, violations)


# In[ ]:


###############################################################################
#  MAIN DRIVER (run_experiment) 
###############################################################################
    # Build grid & marginals
def run_experiment_no_M(solver=None,
    truncation_range=None,
    n=None,
    N=None,
    no_of_options=None,
    short_strikes=None,
    epsilon=None,
    M_bound=None,
    choice=None,
    x0=None,
    seed=136, 
    bounds=None, 
    regularization=None,
    lmda_reg=None                   
):
    """
    Orchestrates everything:
     1) Build discretization points & marginals
     2) Pick short strikes
     3) Compute target call & short calls
     4) Solve min-max
     5) Return final results
    """
    disc_points_new, call_prices_array_new, marg = get_disc_points_and_marginals( truncation_range=truncation_range,\
                                                                                n=n,N=N,choice=choice)
    

    K1 = disc_points_new[0]
    
    K2 = disc_points_new[1]
   
    # Pick short strikes
    
    print(f"Short strikes: {short_strikes}")

    # Compute  short calls

    short_call = get_short_call(T_GRID, T_GRID[1], no_of_options, marg, disc_points_new, short_strikes)
    print(f"length of short_call {len(short_call)}")

    

     
    # pick solver function
    # if solver.lower() == "pulp":
    #     solver_func = solve_max_problem_pulp
    # else:
    solver_func = solve_max_problem_gurobi_without_M

  
    # solve the min_max
    w_opt, final_value, log_results = min_max_optimization(
        solver_func,
        M_bound,
        K1, 
        K2,
        short_strikes,
        no_of_options,
        K,
        marg,
        short_call,
        epsilon,
        x0,
        seed,
        bounds,
        regularization,
        lmda_reg
    )
    
    if w_opt is None:
        # pick the log entry with the smallest objective value
        best = min(log_results, key=lambda d: d["Objective value (outer)"])
        w_opt       = best["Weights"]
        final_value = best["Objective value (outer)"]
        print("Using best point from log; optimiser did not converge.")
    

    # Solve the max problem once more with final_weights
    final_solution = solver_func(
        M_bound,
        K1,
        K2,
        w_opt,
        short_strikes,
        short_call,
        K,
        marg,
        epsilon
    )
    
 

    # Re-compute the target_call
   
    prob_matrix_final = final_solution["p_matrix"]
    target_call = 0
    n_timepoint_1 = len(K1)
    n_timepoint_2 = len(K2)
    
    for i in range(n_timepoint_1):
        for j in range(n_timepoint_2):
            target_call += prob_matrix_final[i][j]* Asian_option_payoff(K1[i],K2[j],K)
    
    hedge_value = w_opt[0] + np.dot(w_opt[1:],short_call)

    # 8) Check constraints using the existing check_constraints_pulp_solution function
   
    if final_solution["Status"] =="Optimal" and final_solution["p_matrix"] is not None:
        all_satisfied, violations = check_constraints_pulp_solution_no_M(
            M_bound,
            K1, K2,
            w_opt,
            short_strikes,
            K,
            marg,
            epsilon,
            final_solution,
            tolerance=1e-8 
        )

        if all_satisfied:
            print("All constraints are satisfied at the final solution!")
        else:
            print("Constraint violations found at the final solution:")
            for v in violations:
                print("  -", v)
    else:
        print("Final solution wasn't optimal, so skipping constraint check.")

   
    # 9) Return final results
    return {
        "w_opt": w_opt,
        "final_value": final_value,
        "log": log_results,
        "N_initial": N, #initial choice of discretization points
        "N": len(K1),#total number of discretization points after interpolation 
        "Value_inner":final_solution["call_value"],
        "p_matrix": final_solution["p_matrix"],
        "target_call_price": target_call,
        "hedge_value": hedge_value
    }


# **FIX THE TRUNCATION RANGE, NUMBER OF DISCRETIZATION POINTS AND SHORT STRIKE RANGE**

# In[28]:


N = 14 #number of discretization points

truncation_range = 2 #truncation range for the discretization points

disc_points_new, call_prices_array_new, marg = get_disc_points_and_marginals( truncation_range=truncation_range,
                                                                            N=N,
                                                                            choice="non-uniform")


#the discretization points
K1, K2 = disc_points_new 

#the list of all short_strikes

short_strike_range = [round(float(disc_points_new[0][7]),4),
                 round(float(disc_points_new[0][6]),4),
                 round(float(disc_points_new[0][8]),4),
                 round(float(disc_points_new[0][5]),4),
                 round(float(disc_points_new[0][9]),4),
                 round(float(disc_points_new[0][4]),4),
                 round(float(disc_points_new[0][11]),4)]

print(f"\n The available short strikes are: { short_strike_range}")


# In[29]:


# #DEFINE A FUNCTION TO PLOT THE CONDITIONAL TARGET OPTION AND HEDGE PORTFOLIO VALUES

# def plot_one_run(K1,K2,N, truncation_range, weights, P_matrix, no_of_options_t1,
#                  run_idx=None,
#                  save=False, save_dir="plots", show=True):
#     """Draw (and optionally save) the target/BS/hedge curves for one run."""
    
   
#     price     = np.zeros(len(K1))
#     BS_price  = np.zeros(len(K1))
#     hedge     = np.zeros(len(K1))
#     Pr_opt    = P_matrix
#     alpha     = marg[0]          
#     tolerance = 1e-17
    
#     # loop over K1 grid
#     for i, k1 in enumerate(K1):
#         #   target conditional price
#         if alpha[i] < tolerance:
#             price[i] = np.nan
#         else:
#             price[i] = sum(
#                 Pr_opt[i, j] * Asian_option_payoff(k1, K2[j], K)
#                 for j in range(len(K2))
#             ) / alpha[i]

#         #   BS proxy
#         BS_price[i] = (
#             0.5 * BS_call(k1, -k1 + 2 * K, T_GRID[1], T, sigma, r)
#             if 2 * K - k1 >= 0
#             else k1 - K
#         )

#         #   hedge value
#         cash = weights[0]
#         wopt = weights[1:]
#         hedge[i] = cash + sum(
#             wopt[l] * max(k1 - short_strike_range[l], 0)
#             for l in range(no_of_options_t1)
#         )

#     # plotting
#     plt.figure(figsize=(5, 3))
#     l=-2
#     plt.plot(K1[2:l], price[2:l], label="target")
#     plt.plot(K1[2:l], BS_price[2:l], label="BS target price")
#     plt.plot(K1[2:l], hedge[2:l], label="hedge portfolio")
#     plt.xlabel("Stock price at short maturity")
#     plt.ylabel("value")
#     plt.title(f"N={N}, trunc={truncation_range}")
#     plt.legend()
#     plt.tight_layout()

#     # ---- 5) save / show
#     if save:
#         import os
#         os.makedirs(save_dir, exist_ok=True)
#         fname = f"plot_N{N}_{run_idx}.png" if run_idx is not None else f"plot_N{N_init}.png"
#         plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
#     if show:
#         plt.show()
#     else:
#         plt.close()


# In[ ]:


def plot_one_run(
        K1, K2,
        truncation_range, N,
        weights, P_matrix,           # hedge weights and optimal joint-prob.
        marg,                        # alpha-vector passed explicitly
        K, T_GRID, sigma, r,         # model constants
        short_strikes,               # array of strikes actually used
        run_idx=None,                # optional run counter (for the file name)
        *,
        save=False,
        folder="plots",              # default folder
        fname=None,                  # default file name pattern
        show=True,
        asian_payoff=None,           # inject the payoff & BS call you use
        bs_call=None,
        lo = None,
        hi = None                    # cut off ends if desired
):
    """
    Draw target, conditional target price and hedge‐portfolio curves.

    Parameters
    ----------
    …
    save        : bool  – write the figure to disk?
    folder      : str|Path – destination directory (created if absent)
    fname       : str|None – file name; if None a default is built
    show        : bool  – show on screen (True) or just save/close (False)
    """

    # ------------------------------------------------------------------ #
    # 1.  Pre-allocate
    # ------------------------------------------------------------------ #
    price    = np.zeros_like(K1, dtype=float)
    bs_price = np.zeros_like(K1, dtype=float)
    hedge    = np.zeros_like(K1, dtype=float)

    cash, wopt = weights[0], np.asarray(weights[1:])
    alpha      = np.asarray(marg[0])
    tol        = 1e-17
    m          = len(short_strikes)

    # ------------------------------------------------------------------ #
    # 2.  Vectorised target & hedge computation
    # ------------------------------------------------------------------ #
    for i, k1 in enumerate(K1):
        if alpha[i] < tol:
            price[i] = np.nan
        else:
            price[i] = (P_matrix[i] *
                        asian_payoff(k1, K2, K)).sum() / alpha[i]

        bs_price[i] = (
            0.5 * bs_call(k1, -k1 + 2*K, T_GRID[1], T_GRID[-1], sigma, r)
            if 2*K - k1 >= 0 else k1 - K
        )

        hedge[i] = cash + np.sum(
            wopt * np.maximum(k1 - short_strikes[:m], 0.0))

    # ------------------------------------------------------------------ #
    # 3.  Plot
    # ------------------------------------------------------------------ #
    # plt.figure(figsize=(5, 3))
    
    # plt.plot(K1[lo:hi], price[lo:hi],    label="Target")
    # plt.plot(K1[lo:hi], bs_price[lo:hi], label="BS price")
    # plt.plot(K1[lo:hi], hedge[lo:hi],    label="Hedge")
    # plt.xlabel("stock price at $t_1$")
    # plt.ylabel("value")
    # plt.title(f"N={N}, No_of_options={no_of_options}")
    # plt.legend()
    # plt.tight_layout()
#causing an error
    
    # ------------------------------------------------------------------ #
    # 4.  Save / show
    # ------------------------------------------------------------------ #
    
    if save:
        from datetime import datetime
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # timestamp: YYYY-MM-DD_HHMMSS
        ts  = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # build a default file name if none supplied
        if fname is None:
            tag  = f"N{N}_run{run_idx}" if run_idx is not None else f"N{N}"
            base = f"hedge_plot_{tag}.png"
        else:
            base = fname if fname.endswith(".png") else f"{fname}.png"

        full_name = f"{ts}_{base}"          # prepend timestamp
        full_path = folder / full_name

        plt.savefig(full_path, dpi=300, bbox_inches="tight")

        # ------------- append a tiny CSV log -----------------------------
        logfile = folder / "plot_index.csv"
        with logfile.open("a") as f:
            # csv columns: timestamp , file_name , run_idx , trunc , N
            f.write(f"{ts},{full_name},{run_idx},{truncation_range},{N}\n")
    if show:
        plt.show()
    else:
        plt.close()


# **SIMULATE STOCK PATHS TO LATER USE FOR PFE CALCULATIONS **

# In[31]:


#### SIMULATE STOCK PATHS TO COMPUTE THE QUANTILES ####
# if __name__ == "__main__" : 
Nsim = 10000
Nsteps = timepoints * 10
delta_t = 1/Nsteps

stock_paths = np.zeros([Nsim,Nsteps+1])
stock_paths[:,0] = stock
np.random.seed(2)
Z1 = np.random.normal(size=[Nsim,Nsteps])

time_indices = [] # list to store the corresponding time-points at which the stock paths are beong calculated
time_indices.append(0)

for i in range(Nsteps):
    stock_paths[:,i+1] = stock_paths[:,i] * np.exp((r - 0.5 * (sigma **2)) * delta_t \
        + sigma * np.sqrt(delta_t) * Z1[:,i])

    time_indices.append(delta_t * (i + 1))

    
index1 = time_indices.index(T_GRID[1]) #index corresponding to short maturity
index2 = time_indices.index(T) #index corresponding to target maturity

print(f"Index corresponding to short maturity {time_indices[index1]} :{index1}")
    
print(f"Index corresponding to target maturity {time_indices[index2]} :{index2}")
    
    #plot the stock paths
t_paths = np.linspace(0, Nsteps, Nsteps+1)
plt.plot(t_paths,stock_paths.transpose())
plt.show()
plt.close()
    
    #print the intermediate time points
print(f"Time indices:\n {time_indices}")
    


# In[32]:


def hedge_payoff_intermediate_time(stock_value,time_point,weights,short_strikes):
    """
     Computes the payoff of the hedge portfolio at intermediate time-points(<= short_maturity) 
    """
    if time_point < T_GRID[1]: # if time_point < short-maturity
        payoff_arrays =  [weights[l+1] * BS_call(stock_value, short_strikes[l], time_point,
                                                                    T_GRID[1], sigma, r) #creates a list of dimension 1000 x len(short_strikes)
                          for l in range(len(short_strikes))]
        payoff_hedge =  weights[0] + np.sum(payoff_arrays, axis = 0)
            
        
        
    if time_point == T_GRID[1]: # if time_point =  short maturity
        payoff_arrays = [weights[l+1] * np.maximum(stock_value - short_strikes[l],0)
                                            for l in range(len(short_strikes))]   
        payoff_hedge =  weights[0] + np.sum(payoff_arrays, axis = 0)
        
    return payoff_hedge
            


# In[33]:


def target_payoff_intermediate_time(stock_value,time_point,T_GRID,target_strike):
    """
     Computes the payoff of the target option at intermediate time-points(less than or equal to the target maturity) 
     """
    target_value = 0
    if time_point < T_GRID[1]:# if time_point < short_maturity
        #first simulate the stock_paths at the short_maturity and T given the current stock price
        np.random.seed(5)
        N_sim_2 = 1000
        N_steps_2 = 2
        
        Z_normal = np.random.normal(size=[N_sim_2,N_steps_2])
        
        #time left to short maturity
        delta_t1 = T_GRID[1] - time_point
        
        stock_paths_initial = stock_value * np.ones(N_sim_2)
        
        #compute the stock prices at short maturity
        stock_paths_T1 = stock_paths_initial * np.exp((r - 0.5 * (sigma **2)) * delta_t1 \
            + sigma * np.sqrt(delta_t1) * Z_normal[:,0])
        
        #time difference between short maturity and target maturity
        delta_t2 = T_GRID[2] - T_GRID[1]
        
        #compute the stock prices at target maturity
        stock_paths_T =  stock_paths_T1 *   np.exp((r - 0.5 * (sigma **2)) * delta_t2 \
            + sigma * np.sqrt(delta_t2) * Z_normal[:,1])
        
        #compute the means
#         print(f"Average of the stock price values:{np.mean(stock_paths_T1),np.mean(stock_paths_T)}")
        
        #compute the payoff at maturity for the stock price paths
        target_value = np.mean([Asian_option_payoff(stock_paths_T1[i], stock_paths_T[i],target_strike) 
                               for i in range(N_sim_2)]) 

    elif time_point == T_GRID[1]: # if time_point = short_maturity
        
        if 2 * target_strike - stock_value >= 0:
            target_value = 0.5 * BS_call(stock_value,-stock_value + 2 * target_strike,T_GRID[1],T,sigma,r)
        else:
            target_value = stock_value - target_strike
    return target_value


# In[34]:


def objective_value_for_stock_paths(time_point,stock_price_values, optimal_weights,T_GRID,K,short_strikes):
    """
    Compute the objective value for the stock price paths at a given time-point
    """
    
    
    
#     print(f"The short strikes are:{short_strikes}")
    
    objective_value = (target_payoff_intermediate_time(stock_price_values ,time_point,T_GRID,K) - 
           hedge_payoff_intermediate_time(stock_price_values, time_point, optimal_weights, short_strikes)
                      )

    return objective_value


# **EXPERIMENT 1: Run the minimization algorithm without bounds on weights**

# In[35]:
no_of_options = 4
short_strikes = short_strike_range[:no_of_options]

# if __name__ == "__main__":
def run_experiment_no_M_without_bounds():
    """
    Runs the no-bound optimization experiment and returns a result dictionary.
    """


    result_data_no_bounds = run_experiment_no_M(
        solver="gurobi",
        truncation_range=truncation_range,
        N=N,
        no_of_options=no_of_options,
        short_strikes=short_strikes,
        epsilon=0,
        M_bound=0,
        choice="non-uniform",
        x0=[0, 1, 0, 0, 0]
    )

    run_dict_no_bounds = {
        "Solver": "gurobi",
        "M_coeff": "No_M",
        "Truncation_range": truncation_range,
        "N_initial": result_data_no_bounds["N_initial"],
        "N": result_data_no_bounds["N"],
        "no_of_options": no_of_options,
        "epsilon": 0.0,
        "Optimal_weights": result_data_no_bounds["w_opt"],
        "Final_value": result_data_no_bounds["final_value"],
        "P_matrix": result_data_no_bounds["p_matrix"],
        "Target_price": result_data_no_bounds["target_call_price"],
        "Hedge_value": result_data_no_bounds["hedge_value"]
    }

    return run_dict_no_bounds

result_data_no_bounds = run_experiment_no_M_without_bounds()
print(f"========== Experiment Complete ==========")

if __name__ == "__main__":
    print("Optimal Weights:", result_data_no_bounds["Optimal_weight"])
    print("Final Value from min-max:", result_data_no_bounds["Final_value"])
        
        
#         current_objective = result_data_no_bounds["final_value"]
        
        
#         if i == 0:
#             previous_objective = current_objective
#             previous_dictionary = run_dict_no_M
#         else:    
#             if current_objective <= previous_objective:
#                 previous_objective = current_objective
#                 previous_dictionary = run_dict_no_M
            
#         list_of_objectives.append(current_objective)
#         list_of_dictionaries.append(run_dict_no_M)


# In[ ]:


#FIX THE FOLDER NAME
folder='BS_Asian_plots_MAY_2025'


# In[ ]:


#PLOT AND CHECK HOW IT LOOKS

Weights_opt_without_bounds = result_data_no_bounds["Optimal_weights"]
P_matrix_no_bounds = result_data_no_bounds["P_matrix"]
if __name__ == "__main__":
    # Weights_opt_without_bounds = result_data_no_bounds["w_opt"]
    # P_matrix_no_bounds = result_data_no_bounds["p_matrix"]
    plot_one_run( K1,
    K2,
    truncation_range,
    N,
    Weights_opt_without_bounds,
    P_matrix_no_bounds,
    marg,
    K,
    T_GRID,
    sigma,
    r,
    short_strikes,
    run_idx=None,
    save=True,
    folder=folder,
    fname='Conditional_plot_no_bounds_zoomed_in',
    show=True,
    asian_payoff=Asian_option_payoff,
    bs_call=BS_call,
    lo = 2,
    hi = -2
)
    


# In[ ]:


# run_dict_no_bounds['Hedge_value']


# In[ ]:


truncation_range =2

N=19
choice ="non-uniform"
no_of_options =2 
disc_points_new, call_prices_array_new, marg = get_disc_points_and_marginals( truncation_range=truncation_range,\
                                                                               N=N,choice=choice)
    

K1 = disc_points_new[0]

K2 = disc_points_new[1]

# Pick short strikes
short_strikes = short_strike_range[:no_of_options] 
    
print(f"Short strikes: {short_strikes}")

# Compute  short calls

short_call = get_short_call(T_GRID, T_GRID[1], no_of_options, marg, disc_points_new, short_strikes)
print(f"length of short_call {len(short_call)}")


# In[ ]:


M_bound = 0
Weights_min_prev = [1] *(no_of_options+1)
result = solve_max_problem_gurobi_without_M(M_bound,
    K1,
    K2,
    Weights_min_prev,
    short_strikes,
    short_call,
    K,
    marg,
    0,)


# In[ ]:


result['P_slack']


# **EXPERIMENT 2: Run the minimization algorithm with bounds on weights**

# In[ ]:


def run_experiment_no_M_with_bounds():
    """
    Runs the min-max experiment with bounds on weights.
    Returns a result dictionary containing weights, matrix, values, etc.
    """
    no_of_options = 4
    short_strikes = short_strike_range[:no_of_options]
    bounds = [(-1.2, 1.2)]

    result_data_with_bounds = run_experiment_no_M(
        solver="gurobi",
        truncation_range=truncation_range,
        N=N,
        no_of_options=no_of_options,
        short_strikes=short_strikes,
        epsilon=0,
        M_bound=0,
        choice="non-uniform",
        x0=[0] * (no_of_options + 1),
        bounds=bounds * (no_of_options + 1)
    )

    run_dict_with_bounds = {
        "Solver": "gurobi",
        "M_coeff": "No_M",
        "Truncation_range": truncation_range,
        "N_initial": result_data_with_bounds["N_initial"],
        "N": result_data_with_bounds["N"],
        "no_of_options": no_of_options,
        "epsilon": 0.0,
        "bounds": bounds,
        "Optimal_weights": result_data_with_bounds["w_opt"],
        "Final_value": result_data_with_bounds["final_value"],
        "P_matrix": result_data_with_bounds["p_matrix"],
        "Target_price": result_data_with_bounds["target_call_price"],
        "Hedge_value": result_data_with_bounds["hedge_value"]
    }

    return run_dict_with_bounds
result_data_with_bounds = run_experiment_no_M_with_bounds()
print(f"========== Experiment Complete ==========")

if __name__ == "__main__":
    print("Optimal Weights:", result_data_with_bounds["Optimal_weight"])
    print("Final Value from min-max:", result_data_with_bounds["Final_value"])
        
        


# In[ ]:


#PLOT AND CHECK HOW IT LOOKS
Weights_opt_with_bounds = result_data_with_bounds["Optimal_weights"]
P_matrix_with_bounds = result_data_with_bounds["P_matrix"]
if __name__ == "__main__":


    plot_one_run( K1,
    K2,
    truncation_range,
    N,
    Weights_opt_with_bounds,
    P_matrix_with_bounds,
    marg,
    K,
    T_GRID,
    sigma,
    r,
    short_strikes,
    run_idx=None,
    save=True,
    folder='BS_Asian_plots_MAY_2025',
    fname='Conditional_plot_with_bounds',
    show=True,         
    asian_payoff=Asian_option_payoff,
    bs_call=BS_call,
#     lo = 2,
#     hi = -2         
)
    
       


# In[ ]:


# run_dict_with_bounds['Hedge_value']


# **EXPERIMENT 3: Run the minimization algorithm with ridge regularization term**

# In[ ]:



regularization = "ridge"
    

    
lmda_reg = 0.043
def run_experiment_with_ridge():
    list_of_objectives = []
    list_of_dictionaries = []
    result_data_with_ridge = run_experiment_no_M(solver="gurobi",
        truncation_range=truncation_range,
        N=N,
        no_of_options=no_of_options,
        short_strikes=short_strikes,
        epsilon= 0,
        M_bound= 0,
        choice="non-uniform",
        x0 = [0] *(no_of_options+1 ),
        regularization = "ridge",
        lmda_reg = lmda_reg                                       
    )



    # Make a similar dictionary
    run_dict_with_ridge = {
        "Solver": "gurobi",
        "M_coeff": "No_M",   # or 0, or None—some label
        "Truncation_range": truncation_range,
        "N_initial": result_data_with_ridge["N_initial"],
        "N": result_data_with_ridge["N"],
        "no_of_options": no_of_options,
        "epsilon": 0.0,
        "regularization_parameter": lmda_reg,
        "Optimal_weights": result_data_with_ridge["w_opt"],
         "Final_value": result_data_with_ridge["final_value"],
         "P_matrix": result_data_with_ridge["p_matrix"],
        "Target_price": result_data_with_ridge["target_call_price"],
        "Hedge_value": result_data_with_ridge["hedge_value"]
     }
    return run_dict_with_ridge
result_data_with_ridge = run_experiment_with_ridge()
print(f"========== Experiment Complete ==========")
if __name__ == "__main__":
    print("Optimal Weights:", result_data_with_ridge["Optima_weight"])
    print("Final Value from min-max:", result_data_with_ridge["Final_value"])
        
        


# In[ ]:


#PLOT AND CHECK HOW IT LOOKS
Weights_opt_with_ridge = result_data_with_ridge["Optimal_weights"]
P_matrix_with_ridge = result_data_with_ridge["P_matrix"]

if __name__ == "__main__":
    plot_one_run( K1,
    K2,
    truncation_range,
    N,
    Weights_opt_with_ridge,
    P_matrix_with_ridge,
    marg,
    K,
    T_GRID,
    sigma,
    r,
    short_strikes,
    run_idx=None,
    save=True,
    folder='BS_Asian_plots_MAY_2025',
    fname='Conditional_plot_with_ridge_zoomed_in',        
    show=True,
    asian_payoff=Asian_option_payoff,
    bs_call=BS_call,
    lo = 2,
    hi = -2
)
    


# In[ ]:


# run_dict_with_ridge['Hedge_value']


# **EXPERIMENT 4: Run the minimization algorithm with lasso regularization term**

# In[ ]:



    

    
lmda_reg = 0.005
def run_experiment_with_lasso():
    list_of_objectives = []
    list_of_dictionaries = []   
    result_data_with_lasso = run_experiment_no_M(solver="gurobi",
        truncation_range=truncation_range,
        N=N,
        no_of_options=no_of_options,
        short_strikes=short_strikes,
        epsilon= 0,
        M_bound= 0,
        choice="non-uniform",
        x0 = [0] *(no_of_options+1 ),
        regularization = "lasso",
        lmda_reg = lmda_reg                                       
    )



    # Make a similar dictionary
    run_dict_with_lasso = {
        "Solver": "gurobi",
        "M_coeff": "No_M",   # or 0, or None—some label
        "Truncation_range": truncation_range,
        "N_initial": result_data_with_lasso["N_initial"],
        "N": result_data_with_lasso["N"],
        "no_of_options": no_of_options,
        "epsilon": 0.0,
        "regularization_parameter": lmda_reg,
        "Optimal_weights": result_data_with_lasso["w_opt"],
         "Final_value": result_data_with_lasso["final_value"],
         "P_matrix": result_data_with_lasso["p_matrix"],
        "Target_price": result_data_with_lasso["target_call_price"],
        "Hedge_value": result_data_with_lasso["hedge_value"]
     }
    return run_dict_with_lasso
result_data_with_lasso = run_experiment_with_lasso()
print(f"========== Experiment Complete ==========")
if __name__ == "__main__":
    print("Optimal Weights:", result_data_with_lasso["Optimal_weights"])
    print("Final Value from min-max:", result_data_with_lasso["Final_value"])
        
        


# In[ ]:


#PLOT AND CHECK HOW IT LOOKS
Weights_opt_with_lasso = result_data_with_lasso["Optimal_weights"]
P_matrix_with_lasso = result_data_with_lasso["P_matrix"]

if __name__ == "__main__":

    plot_one_run( K1,
    K2,
    truncation_range,
    N,
    Weights_opt_with_lasso,
    P_matrix_with_lasso,
    marg,
    K,
    T_GRID,
    sigma,
    r,
    short_strikes,
    run_idx=None,
    save=False,
    folder='BS_Asian_plots_MAY_2025',
    fname='Conditional_plot_with_lasso',        
    show=True,
    asian_payoff=Asian_option_payoff,
    bs_call=BS_call,
#     lo = 2,
#     hi = -2         
)



# In[ ]:


# run_dict_with_lasso['Hedge_value']


# In[ ]:

#store the objective value for each stock path simulation till short maturity i.e. time_indices[index1]
objective_value_list_intermediate_time_no_bounds = np.zeros((index1+1,Nsim))

for i in range(index1+1):
        for j in range(len(stock_paths[:,i])):
             #objective value at the intermediate time points
            objective_value_list_intermediate_time_no_bounds[i,j] = objective_value_for_stock_paths(time_indices[i],
                                                         stock_paths[j,i],
                                                         Weights_opt_without_bounds,
                                                         T_GRID,K,
                                                         short_strikes)
        


# In[ ]:


#store the objective value for each stock path simulation till short maturity i.e. time_indices[index1]
objective_value_list_intermediate_time_with_bounds = np.zeros((index1+1,Nsim))

for i in range(index1+1):
        for j in range(len(stock_paths[:,i])):
             #objective value at the intermediate time points
            objective_value_list_intermediate_time_with_bounds[i,j] = objective_value_for_stock_paths(time_indices[i],
                                                         stock_paths[j,i],
                                                         Weights_opt_with_bounds,
                                                         T_GRID,K,
                                                         short_strikes)


# In[ ]:


#store the objective value for each stock path simulation till short maturity i.e. time_indices[index1]
objective_value_list_intermediate_time_with_ridge = np.zeros((index1+1,Nsim))

for i in range(index1+1):
        for j in range(len(stock_paths[:,i])):
             #objective value at the intermediate time points
            objective_value_list_intermediate_time_with_ridge[i,j] = objective_value_for_stock_paths(time_indices[i],
                                                         stock_paths[j,i],
                                                         Weights_opt_with_ridge,
                                                         T_GRID,K,
                                                         short_strikes)


# In[ ]:


#store the objective value for each stock path simulation till short maturity i.e. time_indices[index1]
objective_value_list_intermediate_time_with_lasso = np.zeros((index1+1,Nsim))

for i in range(index1+1):
        for j in range(len(stock_paths[:,i])):
             #objective value at the intermediate time points
            objective_value_list_intermediate_time_with_lasso[i,j] = objective_value_for_stock_paths(time_indices[i],
                                                         stock_paths[j,i],
                                                         Weights_opt_with_lasso,
                                                         T_GRID,K,
                                                         short_strikes)


# In[ ]:


#list to store the objective values

objective_value_full_list_for_diff_methods = [objective_value_list_intermediate_time_no_bounds,
                                         objective_value_list_intermediate_time_with_bounds,
                                         objective_value_list_intermediate_time_with_ridge,
                                         objective_value_list_intermediate_time_with_lasso]


#list to store the min-max values for the four methods
min_max_list_for_different_methods = [result_data_no_bounds["final_value"],
                                      result_data_with_bounds["final_value"],
                                      result_data_with_lasso["final_value"],
                                      result_data_with_ridge["final_value"]]

#list to store the weights
optimal_weights_list = [Weights_opt_without_bounds,
                        Weights_opt_with_bounds,
                        Weights_opt_with_ridge,
                        Weights_opt_with_lasso]


#list to store the methods used
methods_list = ["no bounds","with bounds", "ridge", "lasso"]


# In[ ]:


#Compute the maximum/peak PFEs with respect to each weight combination till short maturity

peak_PFE_99_list = []
peak_PFE_95_list = []
peak_PFE_5_list = []
peak_PFE_1_list = []
mean_absolute_error_list = []

    
for i,element in enumerate(objective_value_full_list_for_diff_methods):#get the index and the corresponding element from the list
    
    print(f"The optimal weights for {i+1} options are:{optimal_weights_list[i]}")
    
    #print the shape of the the ith element: should be index1 x Nsim
    print(f"Shape of the {i}th element:{element.shape}")
    
    #compute the PFEs at each time point till short maturity
    ninety_nine_percentile_list = np.percentile(element,99, axis=1)
    ninety_fifth_percentile_list = np.percentile(element,95,axis=1) 
    fifth_percentile_list = np.percentile(element,5,axis=1)
    first_percentile_list = np.percentile(element,1,axis=1) 
    mean_of_absolute_error = np.mean(np.abs(element),axis=1) 
    
    #compute the peak PFEs
    peak_PFE_99 = np.max(ninety_nine_percentile_list)
    peak_PFE_95 = np.max(ninety_fifth_percentile_list)
    peak_PFE_5 = np.min(fifth_percentile_list)
    peak_PFE_1 = np.min(first_percentile_list)
    peak_mean_absolute_error = mean_of_absolute_error[-1] 
    
#     print(f"Peak mean absolute error is: {peak_mean_absolute_error}")
#     print(mean_of_absolute_error[-1])
   
    peak_PFE_99_list.append(peak_PFE_99)
    peak_PFE_95_list.append(peak_PFE_95)
    peak_PFE_5_list.append(peak_PFE_5)
    peak_PFE_1_list.append(peak_PFE_1)
    mean_absolute_error_list.append(mean_of_absolute_error[-1])
    
    


# In[ ]:


plt.plot(methods_list,peak_PFE_99_list,label='99 level VAR')
plt.plot(methods_list,peak_PFE_95_list,label='95 level VAR')
plt.plot(methods_list,peak_PFE_5_list,label='5 level VAR')
plt.plot(methods_list,peak_PFE_1_list,label='1 level VAR')
plt.xticks(methods_list)
plt.xlabel("Methods used")
plt.title(f"Peak PFEs till short_maturity {T_GRID[1]}")
plt.legend()
plt.show()
plt.close()


# In[ ]:


#Plot the mean absolute errors and the objective values of the minmax for the four methods
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# data  (already in your workspace)
# methods_list              = ['no bounds', 'with bounds', 'ridge', 'lasso']
# peak_PFE_99_list          = [...]
# peak_PFE_95_list          = [...]
# peak_PFE_5_list           = [...]
# peak_PFE_1_list           = [...]
# --------------------------------------------------------------------

# map each list to a nicer name to loop easily
series = [
    ("Mean Absolute Error", mean_absolute_error_list, "tab:blue"),
    ("Min-Max objective value",min_max_list_for_different_methods , "tab:orange"),
]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6), sharex=True)
axes = axes.ravel()                       # flatten 2×2 → 1-D iterator

for ax, (title, values, colour) in zip(axes, series):
    ax.bar(methods_list, values, color=colour)
    ax.set_title(title)
    ax.set_ylabel("value")
    ax.set_xticklabels(methods_list, rotation=15, ha="right")

fig.suptitle(f"Objective values versus mean absolute errors at short_maturity {T_GRID[1]}")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])   # leave room for suptitle
plt.savefig(f"{folder}/min-max vs MAE till short_maturity {T_GRID[1]}.png", dpi=300, bbox_inches="tight")

# ------------- append a tiny CSV log -----------------------------
logfile = Path(folder) / "plot_index.csv"
plt.show()


# In[ ]:


#PLOT THE PFEs of the four methods

import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# data  (already in your workspace)
# methods_list              = ['no bounds', 'with bounds', 'ridge', 'lasso']
# peak_PFE_99_list          = [...]
# peak_PFE_95_list          = [...]
# peak_PFE_5_list           = [...]
# peak_PFE_1_list           = [...]
# --------------------------------------------------------------------

# map each list to a nicer name to loop easily
series = [
    ("99th percentile", peak_PFE_99_list, "tab:blue"),
    ("95th percentile", peak_PFE_95_list, "tab:orange"),
    ("5th percentile",  peak_PFE_5_list,  "tab:green"),
    ("1st percentile",  peak_PFE_1_list,  "tab:red"),
]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), sharex=True)
axes = axes.ravel()                       # flatten 2×2 → 1-D iterator

for ax, (title, values, colour) in zip(axes, series):
    ax.bar(methods_list, values, color=colour)
    ax.set_title(title)
    ax.set_ylabel("value")
    ax.set_xticklabels(methods_list, rotation=15, ha="right")

fig.suptitle(f"Peak PFEs till short_maturity {T_GRID[1]}")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])   # leave room for suptitle
plt.savefig(f"{folder}/Peak PFEs till short_maturity {T_GRID[1]}.png", dpi=300, bbox_inches="tight")

# ------------- append a tiny CSV log -----------------------------
logfile = Path(folder) / "plot_index.csv"
plt.show()


# In[ ]:


#PLOT THE PFEs of the four methods

import matplotlib.pyplot as plt
from datetime import datetime

# --------------------------------------------------------------------
# data  (already in your workspace)
# methods_list              = ['no bounds', 'with bounds', 'ridge', 'lasso']
# peak_PFE_99_list          = [...]
# peak_PFE_95_list          = [...]
# peak_PFE_5_list           = [...]
# peak_PFE_1_list           = [...]
# --------------------------------------------------------------------

# map each list to a nicer name to loop easily
series = [
    ("99 level VAR", peak_PFE_99_list, "tab:blue"),
    ("95 level VAR", peak_PFE_95_list, "tab:orange"),
]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6), sharex=True)
axes = axes.ravel()                       # flatten 2×2 → 1-D iterator

for ax, (title, values, colour) in zip(axes, series):
    ax.bar(methods_list, values, color=colour)
    ax.set_title(title)
    ax.set_ylabel("value")
    ax.set_xticklabels(methods_list, rotation=15, ha="right")

#Add a time_stamp to prevent accidental overwriting    
 # timestamp: YYYY-MM-DD_HHMMSS
ts  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
fig.suptitle(f"Peak PFEs till short_maturity {T_GRID[1]}")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])   # leave room for suptitle
plt.savefig(f"{folder}/{ts} VAR till short_maturity {T_GRID[1]}.png", dpi=300, bbox_inches="tight")

# ------------- append a tiny CSV log -----------------------------
logfile = Path(folder) / "plot_index.csv"
plt.show()


# In[ ]:


# get_ipython().system('zip -r BS_Asian_plots_MAY_2025.zip BS_Asian_plots_MAY_2025')


# In[ ]:




