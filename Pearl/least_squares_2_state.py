# -*- coding: utf-8 -*-
"""Least_squares_2_state.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IE6JlChSQg6c8U8Cumgsv4pQ4f8qloaw
"""

## Initial Setup

# State/Parameter/Output Dimensions
State_n = 2
Parameter_n = 7
Output_n = 1
Input_n= 7

# Initial Filter stae mean/covariance - as one state
T_ave_ini_model = 22. #room temp
T_wall_ini_model = 24.

#parameters
R_win_ini_model = 9.86
R_w2_ini_model = 9.86
C_in_ini_model = 1000
C_1_ini_model = 1000
C_2_ini_model = 1000
C_3_ini_model = 1000
C_w_ini_model = 1000

# State Covariance
P_model = 1

# Filter process/measurement noise covariances
Q_model = 0.01 #state dynamics-next state uncertainity
R_model = 0.01 #output uncerratinity - always scalor, PQ based on states

# Creating Infinity
Infinity = np.inf


## Creating Optimization Variables

# State Variables
T_ave = SX.sym('T_ave',N+1,1)
T_wall = SX.sym('T_ave',N+1,1)

#Output Variable
y_l = SX.sym('y_l',N,1)

## Getting total time steps
N = y.shape[0]

# Parameter Variables
R_win = SX.sym('R_win',1,1)
R_w2 = SX.sym('R_w2',1,1)
C_in = SX.sym('C_in',1,1)
C_1 = SX.sym('C_1',1,1)
C_2 = SX.sym('C_2',1,1)
C_3 = SX.sym('C_3',1,1)
C_w = SX.sym('C_w',1,1)

# System Matrix
A_matrix = SX.sym('A_matrix',State_n,State_n)

A_matrix[0,0] = (-1/(R_w2*C_in)) - (1/(R_win*C_in))
A_matrix[0,1] = 1/(R_win*C_in)
A_matrix[1,0] = -1/(R_w2*C_in)
A_matrix[1,1] = -2/(R_w2*C_w)

# System Constants
C_matrix = DM(1,State_n)
C_matrix[:,:] = np.reshape(np.array([1,0]), (1,State_n)) #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

#Creating input matrix, i.e B
B_matrix = SX.sym('B_matrix',State_n,Input_n)

B_matrix[0,0] = 0
B_matrix[0,1] = 1/(R_win*C_in)
B_matrix[0,2] = C_1/C_in
B_matrix[0,3] = C_2/C_in
B_matrix[0,4] = 0
B_matrix[0,5] = 1/C_in
B_matrix[0,6] = 1/C_in
B_matrix[1,0] = 1/(R_w2*C_w)
B_matrix[1,1] = 0
B_matrix[1,2] = 0
B_matrix[1,3] = 0
B_matrix[1,4] = C_3/C_w
B_matrix[1,5] = 0
B_matrix[1,6] = 0

## Constructing the Cost Function

# Cost Function Development
CostFunction = 0

## Constructing the Constraints

# Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
T_ave_lb = []
T_ave_ub = []

T_wall_lb = []
T_wall_ub = []

y_lb = []
y_ub = []

R_win_lb = [0]
R_win_ub = [Infinity]
R_w2_lb = [0]
R_w2_ub = [Infinity]
C_in_lb = [0]
C_in_ub = [Infinity]
C_1_lb = [0]
C_1_ub = [Infinity]
C_2_lb = [0]
C_2_ub = [Infinity]
C_3_lb = [0]
C_3_ub = [Infinity]
C_w_lb = [0]
C_w_ub = [Infinity]


Eq_x_lb = []
Eq_y_lb = []

Eq_x_ub = []
Eq_y_ub = []

Eq_x = []
Eq_y = []

# FOR LOOP: For each time step
for ii in range(N):

    # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
    CostFunction += (y_l[ii]-y[ii])**2

    ## State/Covariance Equations - Formulation

    # Creating State Vector
    x_k_1 = SX.sym('x_k_1',State_n,1)
    x_k = SX.sym('x_k',State_n,1)

    x_k_1[0,0] = T_ave[ii+1]
    x_k_1[1,0] = T_wall[ii+1]

    x_k[0,0] = T_ave[ii]
    x_k[1,0] = T_wall[ii]

    #Creating input vector - U

    U_vector = DM(Input_n,1)
    U_vector[:,:] = np.reshape(np.array([T_sol_w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infil[ii]]), (Input_n,1))

    # State Equation
    x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
    y_Eq = C_matrix @ x_k #D matrix only for feed forward system


    # Adding current equations to Equation List
    Eq_x += [x_Eq[1,0]] #[1,0] for 2 state
    Eq_y += [y_Eq[0,0]] #always scalor for all states


    # Adding Equation Bounds, [0,0] for 2 equations
    Eq_x_lb += [0]
    Eq_x_ub += [0]

    Eq_y_lb += [0]
    Eq_y_ub += [0]


    # Adding Variable Bounds
    T_ave_lb += [-Infinity]
    T_ave_ub += [Infinity]

    T_wall_lb += [-Infinity]
    T_wall_ub += [Infinity]

    y_lb += [-Infinity]
    y_ub += [Infinity]

## Adding Variable Bounds - For (N+1)th Variable
T_ave_lb += [-Infinity]
T_ave_ub += [Infinity]

T_wall_lb += [-Infinity]
T_wall_ub += [Infinity]

## Constructing NLP Problem

# Creating Optimization Variable: x
x = vcat([R_win,R_w2,C_in,C_1,C_2,C_3,C_w,T_ave,T_wall,y])

# Creating Cost Function: J
J = CostFunction

# Creating Constraints: g
g = vertcat(*Eq_x, *Eq_y)

# Creating NLP Problem
NLP_Problem = {'f': J, 'x': x, 'g': g}

## Constructiong NLP Solver
NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

## Solving the NLP Problem

# Creating Initial Variables
T_ave_ini = T_ave_ini_model*np.ones((N+1,)).tolist()
T_wall_ini = T_ave_ini_model*np.ones((N+1,)).tolist()
R_win_ini = R_win_ini_model*np.ones((1,)).tolist()
R_w2_ini = R_win_ini_model*np.ones((1,)).tolist()
C_in_ini = C_in_ini_model*np.ones((1,)).tolist()
C_1_ini = C_1_ini_model*np.ones((1,)).tolist()
C_2_ini = C_2_ini_model*np.ones((1,)).tolist()
C_3_ini = C_3_ini_model*np.ones((1,)).tolist()
C_w_ini = C_3_ini_model*np.ones((1,)).tolist()
y_ini = y_ini_model*np.ones((N,)).tolist()

x_initial = vertcat(*R_win_ini, *R_w2_ini, *C_in_ini, *C_1_ini, *C_2_ini, *C_3_ini, *C_w_ini, *T_ave_ini, *T_wall_ini, *y_ini)

# Creating Lower/Upper bounds on Variables and Equations
x_lb = vertcat(*R_win_lb, *R_w2_lb, *C_in_lb, *C_1_lb, *C_2_lb, *C_3_lb, *C_w_lb, *T_ave_lb, *T_wall_lb, *y_lb)

x_ub = vertcat(*R_win_ub, *R_w2_ub, *C_in_ub, *C_1_ub, *C_2_ub, *C_3_ub, *C_w_ub, *T_ave_ub, *T_wall_ub, *y_ub)

G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

# Solving NLP Problem
NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

#----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#

## Getting the Solutions
NLP_Sol = NLP_Solution['x'].full().flatten()

R_win_sol = NLP_Sol[0]
R_w2_sol = NLP_Sol[1]
C_in_sol = NLP_Sol[2]
C_1_sol = NLP_Sol[3]
C_2_sol = NLP_Sol[4]
C_3_sol = NLP_Sol[5]
C_w_sol = NLP_Sol[6]
T_ave_sol = NLP_Sol[7:(N+1)+7]
T_wall_sol = NLP_Sol[(N+1)+7:(N+1)+7+(N+1)]
y_sol = NLP_Sol[(N+1)+7+(N+1):(N+1)+7+(N+1)+N]

## Simulation Plotting
# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting  States
plt.plot(time_vector, T_ave_sol[0:-2], label=r'$\theta$ $(rads)$')
plt.plot(time_vector, y_sol[0:-2], label=r'$\omega = y$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum States - NLP Solution ' + r'x', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Figures
plt.figure()

# Plotting  Parameters - R_win -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, R_win_sol*np.ones((len(time_vector),1)), label=r'$g$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(g)$', fontsize=12)
plt.title('Simple Pendulum Parameter - NLP Solution '+ r'$g$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - C_in -  of Nonlinear System
plt.subplot(222)
plt.plot(time_vector, C_in_sol*np.ones((len(time_vector),1)), label=r'$L$ $(m)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(L)$', fontsize=12)
plt.title('Simple Pendulum Parameter - NLP Solution '+ r'$L$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


# Plotting  Variables - C_1 -  of Nonlinear System
plt.subplot(223)
plt.plot(time_vector, C_1_sol[0:-1], label=r'$S$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(S)$', fontsize=12)
plt.title('Simple Pendulum Variable - NLP Solution '+ r'$S$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


# Plotting  Variables - C_2 -  of Nonlinear System
plt.subplot(224)
plt.plot(time_vector, C_2_sol[0:-1], label=r'$e$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(e)$', fontsize=12)
plt.title('Simple Pendulum Variable - NLP Solution '+ r'$e$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Variables - C_3 -  of Nonlinear System
plt.subplot(224)
plt.plot(time_vector, C_3_sol[0:-1], label=r'$e$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(e)$', fontsize=12)
plt.title('Simple Pendulum Variable - NLP Solution '+ r'$e$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Variables - C_w -  of Nonlinear System
plt.subplot(224)
plt.plot(time_vector, C_w_sol[0:-1], label=r'$e$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(e)$', fontsize=12)
plt.title('Simple Pendulum Variable - NLP Solution '+ r'$e$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Variables - R_w2 -  of Nonlinear System
plt.subplot(224)
plt.plot(time_vector, R_w2_sol[0:-1], label=r'$e$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(e)$', fontsize=12)
plt.title('Simple Pendulum Variable - NLP Solution '+ r'$e$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)