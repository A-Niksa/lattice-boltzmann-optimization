# LIBRARIES
import numpy as np
import math

# SPECIFICATIONS
#                             columns
#         _________________________________________________
#        |                                                |
#        |                                                |
#  rows  |                   ------->                     |
#        |                                                |
#        |________________________________________________|
#                          n_ts, tau
#                  delta_t := 1, delta_x := 1

# VARIABLES
rows = 100 # number of lattice rows
columns = 500 # number of lattice columns
n_ts = 50 # number of timesteps
n_ts_test = 50//10 # number of timesteps in the test sample
tau = 0.6 # collision timescale
n_dir = 9 # number of velocity directions (representative of D2Q9)
c_s = 1/math.sqrt(3) # speed of sound

# LATTICE (D2Q9) DEFINITION
W = np.array([1/36, 1/9, 1/36, 1/9, 4/9, c_s, 1/36, 1/9, 1/36]) # weights
Cx = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1]) # velocity vectors in the x direction
Cy = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]) # velocity vectors in the y direction
X, Y = np.meshgrid(range(columns), range(rows)) # cartesian grid

# INITIAL CONDITIONS
F = np.loadtxt("LBM_initial_conditions.txt").reshape(rows, columns, n_dir)
F_previous_2 = F.copy() # F_previous_2 == F_previous_previous
rho = np.sum(F, 2) # summing F values along axis = 2

# EXTENSIVE FUNCTIONS DEFINITION
A = np.zeros((rows, columns, n_dir, n_ts_test - 1)) # averaging function
D = np.zeros((rows, columns, n_dir, n_ts_test - 1)) # difference function
Omega = np.zeros((rows, columns, n_dir, n_ts_test - 1)) # extensive weight function (different from W)

# CALCULATING F_previous
def LBM_solver(F):
    for i in range(n_dir):
        F[:,:,i] = np.roll(F[:,:,i], Cx[i], axis = 1)
        F[:,:,i] = np.roll(F[:,:,i], Cy[i], axis = 0)
    
    # Variable Updating Step
    rho = np.sum(F, 2)
    Ux = np.sum(F * Cx, 2) / rho # u along the x direction
    Uy = np.sum(F * Cy, 2) / rho # u along the y direction

    # Collision Step
    F_eq = np.zeros(np.shape(F)) # equilibrium velocity distribution function
    for i in range(n_dir):
        F_eq[:,:,i] = W[i] * rho * (1 + (Cx[i] * Ux + Cy[i] * Uy)/(c_s ** 2) + \
            ((Cx[i] * Ux + Cy[i] * Uy) ** 2)/(2 * c_s ** 4) + \
                (Ux * Ux + Uy * Uy)/(2 * c_s ** 2))

    F += -1/tau * (F - F_eq)
    return F
F_previous = LBM_solver(F)

# IMPORTING TEST DATA
F_test = np.loadtxt("LBM_modified_test_F.txt")
F_test = F_test.reshape(rows, columns, n_dir, n_ts_test)

# DRIVING LOOP
for iterator in range(2, n_ts_test - 1):
    F = LBM_solver(F)

    # Extensive Step
    A[:,:,:,iterator] = (F_previous_2 + F_previous) / 2
    D[:,:,:,iterator] = A[:,:,:,iterator] - F_test[:,:,:,iterator] # notice that the size of F_test is different from A or D
    Omega[:,:,:,iterator] = (F - F_previous) / D[:,:,:,iterator]

    # Storing previous F functions to be used in the next iteration
    F_previous_2 = F_previous.copy()
    F_previous = F.copy()

# AVERAGING AND SQUEEZING THE EXTENSIVE WEIGHT FUNCTION (Omega) AND THE DIFFERENCE FUNCTION (D)
Omega_bar = np.zeros((rows, columns, n_dir))
D_bar = np.zeros((rows, columns, n_dir))
for i in range(n_ts_test - 1):
    Omega_bar += Omega[:,:,:,i]
    D_bar += D[:,:,:,i]
Omega_bar /= n_ts_test - 1
D_bar /= n_ts_test - 1


# EXPORTING THE AVERAGE EXTENSIVE WEIGHT FUNCTION (Omega) AND THE AVERAGE DIFFERENCE FUNCTION (D)
np.save("LBM_modified_algo_Omega_avg.npy", Omega_bar)
np.save("LBM_modified_algo_D_avg.npy", D_bar)