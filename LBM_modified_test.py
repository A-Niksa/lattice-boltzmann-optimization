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
F_test = np.zeros((rows, columns, n_dir, n_ts_test))
F_test[:,:,:,1] = F.copy()
rho = np.sum(F, 2) # summing F values along axis = 2

# DRIVING LOOP
for iterator in range(1, n_ts_test):
    # Streaming Step
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
    F_test[:,:,:,iterator] = F.copy()

# EXPORTING DATA
np.savetxt("LBM_modified_test_F.txt", F_test.flatten())