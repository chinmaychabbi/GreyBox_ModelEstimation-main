#importing external model
import numpy as np
from casadi import*

## Initial Setup

# State/Parameter/Output Dimensions
State_n = 1
Parameter_n = 5
Output_n = 1
Input_n =6

# Initial Filter stae mean/covariance
T_ave_ini_model= 9.


R_win_ini_model = 9.86
C_in_ini_model = 1.0
C1_ini_model = 1.0
C2_ini_model = 1.0
C3_ini_model = 1.0

# State Covariance
P_model = 1

# Filter process/measurement noise covariances
Q_model = 0.01
R_model = 0.01

# Creating Infinity
Infinity = np.inf


## Getting total time steps
N = y_measure.shape[0]

## Creating Optimization Variables

# State Variables
T_ave = SX.sym('T_ave',N+1,1)


# Parameter Variables
R_win = SX.sym('R_win',1,1)
C_in = SX.sym('C_in',1,1)
C1 = SX.sym('C1',1,1)
C2 = SX.sym('C2',1,1)
C3 = SX.sym('C3',1,1)

# System Matrix
A_matrix = SX.sym('A_matrix',State_n,State_n)

A_matrix[0,0] = -1/(R_win*C_in)


#Creating input matrix, i.e B
B_matrix = SX.sym('B_matrix',State_n,Input_n)

B_matrix[0,0] = 1/(R_win*C_in)
B_matrix[0,1] = C_1/C_in
B_matrix[0,2] = C_2/C_in
B_matrix[0,3] = C_3/C_in
B_matrix[0,4] = 1/C_in
B_matrix[0,5] = 1/C_in


# Other Variables
S_l = SX.sym('S_l',N,1)
e_l = SX.sym('e_l',N,1)
P_l = SX.sym('P_l',(State_n**2)*(N+1),1)

# System Constants
C_matrix = DM(1,1)
C_matrix [1,1] = np.reshape(np.array([1]), (1,State_n))#np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

R_matrix = DM(Output_n,Output_n)
R_matrix[:,:] = np.reshape(R_model*np.eye(Output_n), (Output_n,Output_n))

Q_matrix = DM(State_n,State_n)
Q_matrix[:,:] = np.reshape(Q_model*np.eye(State_n), (State_n,State_n))

## Constructing the Cost Function

# Cost Function Development
CostFunction = 0

## Constructing the Constraints

# Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
T_ave_lb = []
T_ave_ub = []



R_win_lb = [0]
R_win_ub = [inf]

C_in_lb = [0]
C_in_ub = [inf]

C1_lb = [0]
C1_ub = [inf]

C2_lb = [0]
C2_ub = [inf]

C3_lb = [0]
C3_ub = [inf]

S_lb = []
S_ub = []

e_lb = []
e_ub = []

P_lb = []
P_ub = []


Eq_x_lb = []
Eq_P_lb = []
Eq_S_lb = []
Eq_e_lb = []

Eq_x_ub = []
Eq_P_ub = []
Eq_S_ub = []
Eq_e_ub = []

Eq_x = []
Eq_P = []
Eq_S = []
Eq_e = []

# FOR LOOP: For each time step
for ii in range(N):

    # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
    CostFunction += e_l[ii]**2 * (1/S_l[ii]) + log(S_l[ii])

    ## State/Covariance Equations - Formulation

    # Creating State Vector
    T_ave = SX.sym('T_ave',State_n,1)


    x_k_1[0,0] = T_ave_l[ii+1]


    x_k[0,0] = T_ave_l[ii]

    #Creating Input Vector
    U_vector = DM('Input_m', 1)
    U_vector[:,:] = np.reshape(np.array([T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infill[ii]]), (Input_n, 1))


    # Creating P matrix
    P_matrix_k = SX.sym('P_matrix_k', State_n, State_n)
    P_matrix_k_1 = SX.sym('P_matrix_k_1', State_n, State_n)

    P_matrix_k_1[0,0] = P_l[(ii+1)*State_n**2]


    P_matrix_k[0,0] = P_l[ii*State_n**2]

    # State Equation
    x_Eq = -x_k_1 + A_matrix @ (x_k + P_matrix_k @ C_matrix.T @ (1/S_l[ii]) @ e_l[ii])
    y_Eq = C_matrix @ x_k

    # Covariance Equation
    P_Eq = -P_matrix_k_1 + A_matrix @ (P_matrix_k - P_matrix_k @ C_matrix.T @ (1/S_l[ii]) @ C_matrix @ P_matrix_k) @ A_matrix.T + Q_matrix

    ## Filter Update Equations

    # S_k Equation
    S_Eq = -S_l[ii] + (C_matrix @ P_matrix_k @ C_matrix.T) + R_matrix

    # e_k Equation
    e_Eq = -e_l[ii] + y_measured[ii] - (C_matrix @ x_k)

    # Adding current equations to Equation List
    Eq_x += [x_Eq[0,0], x_Eq[1,0]]
    Eq_y += [y_Eq[0,0]]

    Eq_P += [P_Eq[0,0]]

    Eq_S += [S_Eq]

    Eq_e += [e_Eq]

    # Adding Equation Bounds
    Eq_x_lb += [0, 0]
    Eq_x_ub += [0, 0]

    Eq_y_lb += [0]
    Eq_y_ub += [0]

    Eq_P_lb += [0]
    Eq_P_ub += [0]

    Eq_S_lb += [0]
    Eq_S_ub += [0]

    Eq_e_lb += [0]
    Eq_e_ub += [0]

    # Adding Variable Bounds
    T_ave_lb += [-Infinity]
    T_ave_ub += [Infinity]

    P_lb += [-Infinity]
    P_ub += [Infinity]

    S_lb += [0.0001]
    S_ub += [Infinity]

    e_lb += [-Infinity]
    e_ub += [Infinity]

## Adding Variable Bounds - For (N+1) Variables
T_ave_lb += [-Infinity]
T_ave_ub += [Infinity]


P_lb += [-Infinity]
P_ub += [Infinity]

## Constructing NLP Problem

# Creating Optimization Variable: x
x = vcat([R_win, C_in, C1, C2, C3, T_ave_l, P_l, S_l, e_l])

# Creating Cost Function: J
J = CostFunction

# Creating Constraints: g
g = vertcat(*Eq_x, *Eq_P, *Eq_S, *Eq_e)

# Creating NLP Problem
NLP_Problem = {'f': J, 'x': x, 'g': g}

## Constructiong NLP Solver
NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

## Solving the NLP Problem

# Creating Initial Variables
T_ave_l_ini = (Temp_ave_ini((N+1,))).tolist()

R_win_l_ini = (R_win_ini_model*np.ones((1,))).tolist()
C_in_l_ini = (C_in_ini_model*np.ones((1,))).tolist()
C1_l_ini = (C1_ini_model*np.ones((1,))).tolist()
C2_l_ini = (C2_ini_model*np.ones((1,))).tolist()
C3_l_ini = (C3_ini_model*np.ones((1,))).tolist()

P_l_ini = [1,0,0,1]*(N+1)
S_l_ini = (0.001*np.ones((N,))).tolist()
e_l_ini = np.zeros((N,)).tolist()

x_initial = vertcat(*R_win_ini, *C_in_ini, *C1_in_ini, *C2_in_ini, *C3_in_ini, *T_ave_l_ini, *P_l_ini, *S_l_ini, *e_l_ini)

# Creating Lower/Upper bounds on Variables and Equations
x_lb = vertcat(*R_win_lb, *C_in_lb, *C1_in_lb, *C2_in_lb, *C3_in_lb, *T_ave_lb, *P_lb, *S_lb, *e_lb)

x_ub = vertcat(*R_win_ub, *C_in_ub, *C1_in_ub, *C2_in_ub, *C3_in_ub, *T_ave_ub, *P_ub, *S_ub, *e_ub)

G_lb = vertcat(*Eq_x_lb, *Eq_P_lb, *Eq_S_lb, *Eq_e_lb)

G_ub = vertcat(*Eq_x_ub, *Eq_P_ub, *Eq_S_ub, *Eq_e_ub)

# Solving NLP Problem
NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)


#----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#



## Getting the Solutions
NLP_Sol = NLP_Solution['x'].full().flatten()

R_win_sol = NLP_Sol[0]
C_in_sol = NLP_Sol[1]
C1_in_sol = NLP_Sol[2]
C2_in_sol = NLP_Sol[3]
C3_in_sol = NLP_Sol[4]

T_ave_sol = NLP_Sol[5:(N+1)+5]

P_sol = NLP_Sol[((N+1)+5):(2*(N+1)+5)]
S_sol = NLP_Sol[(2*(N+1)+5):(2*(N+1)+5)+N]
e_sol = NLP_Sol[(2*(N+1)+5)+N:(2*(N+1)+5)+2*N]

## Simulation Plotting
# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting  States
plt.plot(time_vector, T_ave_sol[0:-2], label=r'$\T_ave$ $(rads)$')

plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution ' + r'x', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Figures
plt.figure()

# Plotting  Parameters - R_win -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, R_win_sol*np.ones((len(time_vector),1)), label=r'$R_win$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(R_win)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$R_win$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - C_in -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, C_in_sol*np.ones((len(time_vector),1)), label=r'$C_in$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C_in)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$C_in$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - C1 -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, C1_sol*np.ones((len(time_vector),1)), label=r'$C1$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C1)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$C1$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - C2 -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, C2_sol*np.ones((len(time_vector),1)), label=r'$C2$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C2)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$C2g$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - C3 -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, C3_sol*np.ones((len(time_vector),1)), label=r'$C3$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C3)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$C3$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Variables - P -  of Nonlinear System
plt.subplot(224)
plt.plot(time_vector, P_sol[0:-1], label=r'$P$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(P)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$P$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


# Plotting  Variables - S -  of Nonlinear System
plt.subplot(223)
plt.plot(time_vector, S_sol[0:-1], label=r'$S$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(S)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$S$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


# Plotting  Variables - e -  of Nonlinear System
plt.subplot(224)
plt.plot(time_vector, e_sol[0:-1], label=r'$e$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(e)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$e$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)