# Peter Lie
# AERO 500 / 470

# ICA 4/5

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()


import numpy as np
from scipy.linalg import lu

# Problem 1:
A = np.array([[1, 3, 5],
              [2, 5, 1],
              [2, 3, 8]])
b = np.array([10, 8, 3])

# 1. Invertibility
rank = np.linalg.matrix_rank(A)
print("Problem 1:")
print("Rank of A:", rank)

if rank == A.shape[0]:
    A_inv = np.linalg.inv(A)
    print("A is invertible. Inverse:\n", A_inv)


# 2. Matrix properties
det = np.linalg.det(A)
trace = np.trace(A)
eigvals, eigvecs = np.linalg.eig(A)
norm = np.linalg.norm(A)
lu_piv = lu(A)
svd_vals = np.linalg.svd(A)[1]

print("\nDeterminant:", det, " (looks like floating point issue here again)")
print("Trace:", trace)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)
print("Norm:", norm)
print("LU Decomposition:")
print("P:\n" ,lu_piv[0])
print("L:\n" ,lu_piv[1])
print("U:\n" ,lu_piv[2])
print("Singular values:", svd_vals)


# 3. Solve Ax = b
x = np.linalg.solve(A, b)
print("Solution to Ax = b:\n", x)



# Problem 2:

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

tol = 1e-8 # Tolerancing for accuracy

def van_der_pol(t, state, mu):
    x, dx = state
    ddx = mu * (1 - x**2) * dx - x
    return [dx, ddx]

mu = 1
t_span = (0, 10)

# Mesh grid of initial conditions
x0_vals = np.linspace(-4, 4, 10)  # 9 points from -4 to 4
dx0_vals = np.linspace(-4, 4, 10)
initial_conditions = [(x0, dx0) for x0 in x0_vals for dx0 in dx0_vals]


# Plotting trajectories
plt.figure(figsize=(10, 6))

for x0, dx0 in initial_conditions:
    if x0**2 + dx0**2 > 2.5:
        sol = solve_ivp(van_der_pol, t_span, [x0, dx0], args=(mu,), rtol = tol, atol = tol)
        plt.plot(sol.y[0], sol.y[1])
    else:
        t_span = t_span = (0, 3)
        sol = solve_ivp(van_der_pol, t_span, [x0, dx0], args=(mu,), rtol = tol, atol = tol)
        plt.plot(sol.y[0], sol.y[1])

plt.title("VDP Oscillator Phase Plot (${\\mu}$ = 1)")
plt.xlabel("x(t)")
plt.ylabel("xdot(t)")
plt.grid(True)
plt.show()


