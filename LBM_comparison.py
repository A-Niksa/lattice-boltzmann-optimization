# LIBRARIES
import subprocess
import time
import numpy as np

# FUNCTIONS
def prerequisiteFiles():
#    subprocess.call(fname_preliminary, shell = True)
    subprocess.call(fname_modified_test, shell = True)
    subprocess.call(fname_modified_algo, shell = True)

def preliminaryRun():
    subprocess.call(fname_preliminary, shell = True)

def modifiedRun():
    subprocess.call(fname_modified, shell = True)

def fileTimer(f):
    start_time = time.time()
    f()
    end_time = time.time()
    return end_time - start_time

def similarityTester():
    F_pre = np.load("LBM_preliminary_F_EXPRT.npy")
    F_mod = np.load("LBM_modified_F_EXPRT.npy")
    I,J,K,W = np.shape(F_pre)
    simil = 0
    for i in range(W):
        Val_bin = (F_mod[:,:,:,i] - F_pre[:,:,:,i]) / F_pre[:,:,:,i]
        simil += np.sum(np.sum(np.sum(Val_bin, 2),1),0)
    return 1-abs(simil/(W*I*J*K))
        

# FILENAMES
fname_preliminary = "LBM_preliminary.py"
fname_modified = "LBM_modified.py"
fname_modified_test = "LBM_modified_test.py"
fname_modified_algo = "LBM_modified_algo.py"

# PREPARING THE PREREQUISITE FILES
prerequisiteFiles()

# BENCHMARKING PRELIMINARY AND MODIFIED RUNS
t_preliminary = fileTimer(preliminaryRun); str_preliminary = "Preliminary Run: %s s" % t_preliminary
t_modified = fileTimer(modifiedRun); str_modified = "Modified Run: %s s" % t_modified
print(str_preliminary)
print(str_modified)

# EXPORTING BENCHMARKING DATA
str_arr = np.array([t_preliminary,t_modified])
np.savetxt("LBM_comparison_results.txt",str_arr)

# SIMILARITY TESTING
simil_percent = similarityTester() * 100
print("Similarity (%) :", simil_percent)