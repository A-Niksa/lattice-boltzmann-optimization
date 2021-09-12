# LIBRARIES
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import glob
import cv2
import os
import time

# FUNCTIONS
def varUpdate(F): # Variable Updating Step
    rho = np.sum(F, 2)
    Ux = np.sum(F * Cx, 2) / rho # u along the x direction
    Uy = np.sum(F * Cy, 2) / rho # u along the y direction
    return rho, Ux, Uy

def F_Rolling(F):
    for i in range(n_dir):
        F[:,:,i] = np.roll(F[:,:,i], Cx[i], axis = 1)
        F[:,:,i] = np.roll(F[:,:,i], Cy[i], axis = 0)

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
F_previous = F.copy()
rho = np.sum(F, 2) # summing F values along axis = 2

# IMPORTING Omega AND D
Omega_avg = np.load("LBM_modified_algo_Omega_avg.npy")
D_avg = np.load("LBM_modified_algo_D_avg.npy")
Omega_D = Omega_avg * D_avg

# DEFINING F_EXPRT
F_EXPRT = np.zeros((rows, columns, n_dir, n_ts))

# DRIVING LOOP
for iterator in range(n_ts):
    if iterator % 3 == 0:
        ##t0 = time.time()
        F_Rolling(F)
        rho, Ux, Uy = varUpdate(F)
        F += Omega_D
        ##t1 = time.time()
        ##print("mod",t1-t0)

    else:
        ##t0 = time.time()
        # Streaming Step
        F_Rolling(F)

        # Variable Updating Step
        rho, Ux, Uy = varUpdate(F)

        # Collision Step
        F_eq = np.zeros(np.shape(F)) # equilibrium velocity distribution function
        for i in range(n_dir):
            F_eq[:,:,i] = W[i] * rho * (1 + (Cx[i] * Ux + Cy[i] * Uy)/(c_s ** 2) + \
                ((Cx[i] * Ux + Cy[i] * Uy) ** 2)/(2 * c_s ** 4) + \
                    (Ux * Ux + Uy * Uy)/(2 * c_s ** 2))

        F += -1/tau * (F - F_eq)
        F_EXPRT[:,:,:,iterator] = F.copy()
        ##t1 = time.time()
        ##print("pre",t1-t0)

    # Visualization
    i_modified = iterator + 1 # so that the numbered title starts from 1 and ends in n
    vorticity = (np.roll(Ux, -1, axis=0) - np.roll(Ux, 1, axis=0)) - (np.roll(Uy, -1, axis=1) - np.roll(Uy, 1, axis=1))
    plt.imshow(vorticity, cmap = 'bwr')
    plt.title("%s" % i_modified)
    plt.savefig("LBM_modified_%s" % i_modified)

# VIDEO EXPORTATION
test_img = cv2.imread("LBM_modified_1.png")
height, width, layers = test_img.shape
framesize = (width, height)
output = cv2.VideoWriter("LBM_modified_video.avi",cv2.VideoWriter_fourcc(*'DIVX'),10,framesize)
for fname in sorted(glob.glob("*.png"), key = os.path.getmtime):
    img = cv2.imread(fname)
    output.write(img)
    os.remove(fname)

output.release()

# ARRAY EXPORTATION FOR COMPARISON
np.save("LBM_modified_F_EXPRT.npy",F_EXPRT)