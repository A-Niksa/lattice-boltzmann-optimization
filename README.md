# Lattice-Boltzmann-Optimization
This project attempts to decrease the computational cost and thus, the runtime of LBM simulations. The optimization method has been explained in algorithm_explanation.pdf.

WHAT DOES EACH FILE DO?
* LBM_preliminary.py: Uses the BGK model of LBM (standard method).
* LBM_modified_test.py: Runs a shorter version of LBM_preliminary for sampling purposes.
* LBM_modified_algo.py: Does the same as LBM_modified_test.py + processing the results for creating arrays which will help us with increasing speed
* LBM_modified.py: Takes the arrays produced by LBM_modified_algo.py and at a few select steps, uses another iterative equation instead of the BGK model which can decrease computational cost.
* LBM_comparison.py: Compares runtimes of LBM_preliminary.py and LBM_modified.py and calculates the approximate similarity of their results as well.

HOW TO RUN EACH FILE?
* LBM_preliminary.py: Independent. Can be run directly.
* LBM_modified_test.py: Independent. Can be run directly.
* LBM_modified_algo.py: Run after running LBM_modified_test.py
* LBM_modified.py: Run after running LBM_modified_algo.py
* LBM_comparison.py: Independent. Can be run directly.
