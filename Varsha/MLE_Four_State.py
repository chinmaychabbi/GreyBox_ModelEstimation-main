#!pip install casadi

#importing external model
import numpy as np
from casadi import*

## Initial Setup

# State/Parameter/Output Dimensions
State_n = 4
Parameter_n = 12
Output_n = 1
Input_n = 8

# Initial Filter stae mean/covariance
T_ave_ini_model = 9.
T_wall_ini_model = 0.5
T_attic_ini_model = 9.
T_im_ini_model = 0.5

R_win_ini_model = 9.86
R_(w/2)_ini_model = 9.86
R_attic_ini_model = 1.0
R_im_ini_model = 9.86
R_roof_ini_model = 1.0
C_in_ini_model = 1.0
C_w_ini_model = 1.0
C_attic_ini_model = 9.86
C_im_ini_model = 9.86
C1_ini_model = 1.0
C2_ini_model = 1.0
C3_ini_model = 1.0

# State Covariance
P_model = 4

# Filter process/measurement noise covariances
Q_model = 0.01
Q_params = 0.01
R_model = 0.01

# Creating Infinity
Infinity = np.inf

## Getting total time steps
N = y_measure.shape[1]

## Creating Optimization Variables

# State Variables
T_ave = SX.sym('T_ave',N+1,1)
T_wall = SX.sym('T_wall',N+1,1)
T_attic = SX.sym('T_attic',N+1,1)
T_im = SX.sym('T_im',N+1,1)


# Parameter Variables
R_win = SX.sym('R_win',1,1)
R_w/2 = SX.sym('R_w/2',1,1)
R_attic = SX.sym('R_attic',1,1)
R_im = SX.sym('R_im',1,1)
R_roof = SX.sym('R_roof',1,1)

C_in = SX.sym('C_in',1,1)
C_w = SX.sym('C_w',1,1)
C_attic = SX.sym('C_attic',1,1)
C_im = SX.sym('C_im',1,1)
C1 = SX.sym('C1',1,1)
C2 = SX.sym('C2',1,1)
C3 = SX.sym('C3',1,1)

# System Matrix
A_matrix = SX.sym('A_matrix',State_n,State_n)

A_matrix[0,0] = (-1/C_in(1/(R_w/2) + 1/R_attic + 1/R_im + 1/R_win))
A_matrix[0,1] = 1/(C_in*(R_w/2))
A_matrix[0,2] = 1/(C_in*R_attic)
A_matrix[0,3] = 1/(C_in*R_im)
A_matrix[1,0] = 1/(C_w*(R_w/2))
A_matrix[1,1] = -2/(C_w*(R_w/2))
A_matrix[1,2] = 0
A_matrix[1,3] = 0
A_matrix[2,0] = 1/(C_attic*R_attic)
A_matrix[2,1] = 0
A_matrix[2,2] = -2/(C_attic*R_attic)
A_matrix[2,3] = 0
A_matrix[3,0] = 1/(C_im*R_im)
A_matrix[3,1] = 0
A_matrix[3,2] = 0
A_matrix[3,3] = -1/(C_im*R_im)


#Creating input matrix, i.e B
B_matrix = SX.sym('B_matrix',State_n,Input_m)

B_matrix[0,0] = t_s * (0)
B_matrix[0,1] = t_s * (0)
B_matrix[0,2] = t_s * 1/(C_in*R_win)
B_matrix[0,3] = t_s * C1/C_in
B_matrix[0,4] = t_s * C2/C_in
B_matrix[0,5] = t_s * (0)
B_matrix[0,6] = t_s * (1/C_in)
B_matrix[0,7] = t_s * (1/C_in)


B_matrix[1,0] = t_s * (0)
B_matrix[1,1] = t_s * (1/C_w*R_(w/2))
B_matrix[1,2] = t_s * (0)
B_matrix[1,3] = t_s * (0)
B_matrix[1,4] = t_s * (0)
B_matrix[1,5] = t_s * (0)
B_matrix[1,6] = t_s * (0)
B_matrix[1,7] = t_s * (0)

B_matrix[2,0] = t_s * (1/(C_attic*R_roof) )
B_matrix[2,1] = t_s * (0)
B_matrix[2,2] = t_s * (0)
B_matrix[2,3] = t_s * (0)
B_matrix[2,4] = t_s * (0)
B_matrix[2,5] = t_s * (0)
B_matrix[2,6] = t_s * (0)
B_matrix[2,7] = t_s * (0)


B_matrix[3,0] = t_s * (0)
B_matrix[3,1] = t_s * (0)
B_matrix[3,2] = t_s * (0)
B_matrix[3,3] = t_s * (0)
B_matrix[3,4] = t_s * (0)
B_matrix[3,5] = t_s * (C3/C_im)
B_matrix[3,6] = t_s * (0)
B_matrix[3,7] = t_s * (0)


# Other Variables
S_l = SX.sym('S_l',N,1)
e_l = SX.sym('e_l',N,1)
P_l = SX.sym('P_l',(State_n**State_n)*(N+1),1)

# System Constants
C_matrix = DM(1,State_n)
C_matrix[:,:] = np.reshape(np.array([1,0,0,0]), (1,State_n))


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

T_wall_lb = []
T_wall_ub = []

T_attic_lb = []
T_attic_ub = []

T_im_lb = []
T_im_ub = []

R_win_lb = [0]
R_win_ub = [Infinity]

R_attic_lb = [0]
R_aattic_ub = [Infinity]

R_im_lb = [0]
R_im_ub = [Infinity]

R_roof_lb = [0]
R_roof_ub = [Infinity]


C_in_lb = [0]
C_in_ub = [Infinity]

C_w_lb = [0]
C_w_ub = [Infinity]

C_attic_lb = [0]
C_attic_ub = [Infinity]

C_im_lb = [0]
C_im_ub = [Infinity]

C1_lb = [0]
C1_ub = [inf]

C2_lb = [0]
C2_ub = [Infinity]

C3_lb = [0]
C3_ub = [Infinity]



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
    T_wall = SX.sym('T_wall',State_n,1)
    T_aattic = SX.sym('T_attic',State_n,1)
    T_im = SX.sym('T_im',State_n,1)
    
    x_k_1[0,0] = T_ave_l[ii+1]
    x_k_1[1,0] = T_wall_l[ii+1]
    x_k_1[2,0] = T_attic_l[ii+1]
    x_k_1[3,0] = T_im_l[ii+1]

    x_k[0,0] = T_ave_l[ii]
    x_k[1,0] = T_wall_l[ii]
    x_k[2,0] = T_attic_l[ii]
    x_k[3,0] = T_im_l[ii]

    #Creating Input Vector
    U_vector = DM('Input_n', 1)
    U_vector[:,:] = np.reshape(np.array([T_sol,r[ii], T_sol,w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infill[ii]]), ('Input_n', 1))
   
    
    
    # Creating P matrix
    P_matrix_k = SX.sym('P_matrix_k', State_n, State_n)
    P_matrix_k_1 = SX.sym('P_matrix_k_1', State_n, State_n)
    
    P_matrix_k_1[0,0] = P_l[(ii+1)*State_n**2]
    P_matrix_k_1[0,1] = P_l[((ii+1)*State_n**2)+1]
    P_matrix_k_1[0,2] = P_l[((ii+1)*State_n**2)+2]
    P_matrix_k_1[0,3] = P_l[((ii+1)*State_n**2)+3]
    P_matrix_k_1[1,0] = P_l[((ii+1)*State_n**2)+4]
    P_matrix_k_1[1,1] = P_l[((ii+1)*State_n**2)+5]
    P_matrix_k_1[1,2] = P_l[((ii+1)*State_n**2)+6]
    P_matrix_k_1[1,3] = P_l[((ii+1)*State_n**2)+7]
    P_matrix_k_1[2,0] = P_l[((ii+1)*State_n**2)+8]
    P_matrix_k_1[2,1] = P_l[((ii+1)*State_n**2)+9]
    P_matrix_k_1[2,2] = P_l[((ii+1)*State_n**2)+10]
    P_matrix_k_1[2,3] = P_l[((ii+1)*State_n**2)+11]
    P_matrix_k_1[3,0] = P_l[((ii+1)*State_n**2)+12]
    P_matrix_k_1[3,1] = P_l[((ii+1)*State_n**2)+13]
    P_matrix_k_1[3,2] = P_l[((ii+1)*State_n**2)+14]
    P_matrix_k_1[3,3] = P_l[((ii+1)*State_n**2)+15]
    
    
    P_matrix_k[0,0] = P_l[ii*State_n**2]
    P_matrix_k[0,1] = P_l[(ii*State_n**2)+1]
    P_matrix_k[0,2] = P_l[(ii*State_n**2)+2]
    P_matrix_k[0,3] = P_l[(ii*State_n**2)+3]
    P_matrix_k[1,1] = P_l[(ii*State_n**2)+4]
    P_matrix_k[1,1] = P_l[(ii*State_n**2)+5]
    P_matrix_k[1,2] = P_l[(ii*State_n**2)+6]
    P_matrix_k[1,3] = P_l[(ii*State_n**2)+7]
    P_matrix_k[2,0] = P_l[(ii*State_n**2)+8]
    P_matrix_k[2,1] = P_l[(ii*State_n**2)+9]
    P_matrix_k[2,2] = P_l[(ii*State_n**2)+10]
    P_matrix_k[2,3] = P_l[(ii*State_n**2)+11]
    P_matrix_k[3,0] = P_l[(ii*State_n**2)+12]
    P_matrix_k[3,1] = P_l[(ii*State_n**2)+13]
    P_matrix_k[3,2] = P_l[(ii*State_n**2)+14]
    P_matrix_k[3,3] = P_l[(ii*State_n**2)+15]
    
    # State Equation
    x_Eq = -x_k_1 + A_matrix @ (x_k + P_matrix_k @ C_matrix.T @ (1/S_l[ii]) @ e_l[ii])

    # Covariance Equation
    P_Eq = -P_matrix_k_1 + A_matrix @ (P_matrix_k - P_matrix_k @ C_matrix.T @ (1/S_l[ii]) @ C_matrix @ P_matrix_k) @ A_matrix.T + Q_matrix

    ## Filter Update Equations
    
    # S_k Equation
    S_Eq = -S_l[ii] + (C_matrix @ P_matrix_k @ C_matrix.T) + R_matrix
    
    # e_k Equation
    e_Eq = -e_l[ii] + y_measured[0,ii] - (C_matrix @ x_k)

    # Adding current equations to Equation List
    Eq_x += [x_Eq[0,0], x_Eq[0,1], x_Eq[1,0], x_Eq[1,1]]
    
    Eq_P += [P_Eq[0,0], P_Eq[0,1], P_Eq[0,2], P_Eq[0,3]
             P_Eq[1,0], P_Eq[1,1], P_Eq[1,2], P_Eq[1,3]
             P_Eq[2,0], P_Eq[2,1], P_Eq[2,2], P_Eq[2,3]
             P_Eq[3,0], P_Eq[3,1], P_Eq[3,2], P_Eq[3,3]]
    
    Eq_S += [S_Eq]
    
    Eq_e += [e_Eq]

    # Adding Equation Bounds
    Eq_x_lb += [0, 0, 0, 0]
    Eq_x_ub += [0, 0, 0, 0]

    Eq_y_lb += [0, 0, 0, 0]
    Eq_y_ub += [0, 0, 0, 0]
    
    Eq_P_lb += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Eq_P_ub += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    Eq_S_lb += [0]
    Eq_S_ub += [0]
    
    Eq_e_lb += [0]
    Eq_e_ub += [0]

    # Adding Variable Bounds
    T_ave_lb += [-Infinity]
    T_ave_ub += [Infinity]

    T_wall_lb += [-Infinity]
    T_wall_ub += [Infinity]
    
    P_lb += [-Infinity, -Infinity, -Infinity, -Infinity
             -Infinity, -Infinity, -Infinity, -Infinity
             -Infinity, -Infinity, -Infinity, -Infinity
             -Infinity, -Infinity, -Infinity, -Infinity]
    P_ub += [Infinity, Infinity, Infinity, Infinity
             Infinity, Infinity, Infinity, Infinity
             Infinity, Infinity, Infinity, Infinity
             Infinity, Infinity, Infinity, Infinity]

    S_lb += [0.0001]
    S_ub += [Infinity]

    e_lb += [-Infinity]
    e_ub += [Infinity]

## Adding Variable Bounds - For (N+1) Variables
T_ave_lb += [-Infinity]
T_ave_ub += [Infinity]

T_wall_lb += [-Infinity]
T_wall_ub += [Infinity]

P_lb += [-Infinity, -Infinity, -Infinity, -Infinity
         -Infinity, -Infinity, -Infinity, -Infinity
         -Infinity, -Infinity, -Infinity, -Infinity
         -Infinity, -Infinity, -Infinity, -Infinity]
P_ub += [Infinity, Infinity, Infinity, Infinity
         Infinity, Infinity, Infinity, Infinity
         Infinity, Infinity, Infinity, Infinity
         Infinity, Infinity, Infinity, Infinity]    

## Constructing NLP Problem

# Creating Optimization Variable: x
x = vcat([R_win, R_w/2, R_attic, R_im, R_roof, C_in, C_w, C_attic, C_im, C1, C2, C3, T_ave_l, T_wall_l, T_attic_l, T_im_l, P_l, S_l, e_l])

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
T_ave_l_ini = (Temp_ave_int((N+1,))).tolist()
T_wall_l_ini = (Temp_wall_int((N+1,))).tolist()
T_attic_l_ini = (Temp_ave_int((N+1,))).tolist()
T_im_l_ini = (Temp_wall_int((N+1,))).tolist()

R_win_l_ini = (R_win_ini_model*np.ones((1,))).tolist()
R_w/2_l_ini = (R_w/2_ini_model*np.ones((1,))).tolist()
R_attic_l_ini = (R_win_ini_model*np.ones((1,))).tolist()
R_im_l_ini = (R_w/2_ini_model*np.ones((1,))).tolist()
R_roof_l_ini = (R_win_ini_model*np.ones((1,))).tolist()

C_in_ini = (C_in_ini_model*np.ones((1,))).tolist()
C_w_ini = (C_w_ini_model*np.ones((1,))).tolist()
C_attic_ini = (C_in_ini_model*np.ones((1,))).tolist()
C_im_ini = (C_w_ini_model*np.ones((1,))).tolist()
C1_ini = (C1_ini_model*np.ones((1,))).tolist()
C2_ini = (C2_ini_model*np.ones((1,))).tolist()
C3_ini = (C3_ini_model*np.ones((1,))).tolist()


P_l_ini = [1,0,0,1]*(N+1)
S_l_ini = (0.001*np.ones((N,))).tolist()
e_l_ini = np.zeros((N,)).tolist()

x_initial = vertcat(*R_win_ini, *R_w/2_ini, *R_attic_ini, *R_im_ini, *R_roof_ini, *C_in_ini, *C_w_ini, *C_attic_ini, *C_im_ini, *C1_ini, *C2_ini, *C3_ini *T_ave_l_ini, *T_wall_l_ini, T_attic_l_ini, *T_im_l_ini, *P_l_ini, *S_l_ini, *e_l_ini)

# Creating Lower/Upper bounds on Variables and Equations
x_lb = vertcat(*R_win_lb, *R_w/2_lb, *R_attic_lb, *R_im_lb, *R_roof_lb, *C_in_lb, *C_w_lb, *C_attic_lb, *C_im_lb, *C1_lb, *C2_lb, *C3_lb, *T_ave_lb, *T_wall_lb, *P_lb, *S_lb, *e_lb)

x_ub = vertcat(*R_win_ub, *R_w/2_ub, *R_attic_ub, *R_im_ub, *R_roof_ub, *C_in_ub, *C_w_ub, *C_attic_ub, *C_im_ub, *C1_ub, *C2_ub, *C3_ub, *T_ave_ub, *T_wall_ub, *P_ub, *S_ub, *e_ub)

G_lb = vertcat(*Eq_x_lb, *Eq_P_lb, *Eq_S_lb, *Eq_e_lb)

G_ub = vertcat(*Eq_x_ub, *Eq_P_ub, *Eq_S_ub, *Eq_e_ub)

# Solving NLP Problem
NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)


#----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#

## Getting the Solutions
NLP_Sol = NLP_Solution['x'].full().flatten()

R_win_sol = NLP_Sol[0]
R_w/2_sol = NLP_Sol[1]
R_attic_sol = NLP_Sol[2]
R_im_sol = NLP_Sol[3]
R_roof_sol = NLP_Sol[4]
C_in_sol = NLP_Sol[5]
C_w_sol = NLP_Sol[6]
C_attic_sol = NLP_Sol[7]
C_im_sol = NLP_Sol[8]
C1_sol = NLP_Sol[9]
C2_sol = NLP_Sol[10]
C3_sol = NLP_Sol[11]


T_ave_sol = NLP_Sol[12:(N+1)+12]
T_wall_sol = NLP_Sol[((N+1)+12):(2*(N+1)+12)]
T_attic_sol = NLP_Sol[(2*(N+1)+12):(2*(N+1)+12)+N]
T_im_sol = NLP_Sol[(2*(N+1)+12)+N:(2*(N+1)+12)+2*N]

P_sol = NLP_Sol[(2*(N+1)+12)+2*N:(2*(N+1)+12)+2*2*N]
S_sol = NLP_Sol[(2*(N+1)+12)+2*2*N:(2*(N+1)+12)+2*2*2*N]
e_sol = NLP_Sol[(2*(N+1)+12)+2*2*2*N:(2*(N+1)+12)+2*2*2*2*N]


## Simulation Plotting
# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting  States
plt.plot(time_vector, T_ave_sol[0:-2], label=r'$\T_ave$ $(rads)$')
plt.plot(time_vector, T_wall_sol[0:-2], label=r'$\T_wall$ $(rads)$')
plt.plot(time_vector, T_attic_sol[0:-2], label=r'$\T_attic$ $(rads)$')
plt.plot(time_vector, T_im_sol[0:-2], label=r'$\T_im$ $(rads)$')
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

# Plotting  Parameters - R_w/2 -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, R_w/2_sol*np.ones((len(time_vector),1)), label=r'$R_w/2$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(R_w/2)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$R_w/2$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - R_attic -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, R_attic_sol*np.ones((len(time_vector),1)), label=r'$R_attic$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(R_attic)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$R_attic$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - R_im -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, R_im_sol*np.ones((len(time_vector),1)), label=r'$R_im$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(R_im)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$R_im$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - R_roof -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, R_roof_sol*np.ones((len(time_vector),1)), label=r'$R_roof$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(R_roof)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$R_roof$', fontsize=14)
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

# Plotting  Parameters - C_w -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, C_w_sol*np.ones((len(time_vector),1)), label=r'$C_w$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C_w)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$C_w$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - C_attic -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, C_attic_sol*np.ones((len(time_vector),1)), label=r'$C_attic$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C_attic)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$C_attic$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - C_im -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, C_im_sol*np.ones((len(time_vector),1)), label=r'$C_im$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(C_im)$', fontsize=12)
plt.title('Maximum LikelyHood Estimation - NLP Solution '+ r'$C_im$', fontsize=14)
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



