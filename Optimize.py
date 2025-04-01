# Peter Lie
# AERO 557 Advanced Orbital Mechanics
# Functions for orbit optimization in Python
# As each function needs to be created specifically for the problem statement, 
# included functions match up with certain homework problems


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# For 2024 Homework 3, Problem 3:
# Two dimensional motion to end up at y = 1 while maximizing x velocity


# Define the equations of motion:
# Differential equations for 2D motion
# Need as many adjoints as variables in the state

def steering_eq(t, state):
    x, y, xdot, ydot, lambda1, lambda2, lambda3, lambda4 = state
  
    lambdav = np.sqrt(lambda3**2 + lambda4**2)

    # Equations of motion
    xddot = - lambda3 / lambdav
    yddot = - lambda4 / lambdav

    # Adjoint equations
    lambda1_dot = 0
    lambda2_dot = 0
    lambda3_dot = -lambda1
    lambda4_dot = -lambda2
    
    return [xdot, ydot, xddot, yddot, lambda1_dot, lambda2_dot, lambda3_dot, lambda4_dot]

# Fsolve equations:
def steering_eq_fsolve(init_state, t_span):

    # Default tolerance for ODE and optimization
    tol = 1e-9

    # Propagate the state with the given initial conditions
    sol = solve_ivp(steering_eq, t_span, init_state, rtol=tol, atol=tol)
    final_state = sol.y[:, -1]  # Final state

    # Fsolve constraints: 
    # These equations are what Fsolve sends to 0 to comply with the state and adjoint requirements
    # Final constraints
    final_y = final_state[1] - 1  # Final height = 1
    final_ydot = final_state[3]   # Final vertical velocity = 0
    final_lambda1 = final_state[4] # Final lambda1 = 0
    final_lambda3 = final_state[6] + 1 # Final lambda3 = -1
    # Initial constraints
    initial_state1 = init_state[0]  # Initial positions and velocities are 0
    initial_state2 = init_state[1]  # Initial positions and velocities are 0
    initial_state3 = init_state[2]  # Initial positions and velocities are 0
    initial_state4 = init_state[3]  # Initial positions and velocities are 0

    return [final_y, final_ydot, final_lambda1, final_lambda3, initial_state1, initial_state2, initial_state3, initial_state4]


# Solving equations:

# Set tolerance
tol = 1e-9

# Set time span
t_span = [0, 3]

# Initial guesses
# State starts at origin with zero speed, adjoint guesses at 0 or -1
init_guess = [0, 0, 0, 0, 0, -1, -1, -1]

# First propagation with initial guess
sol1 = solve_ivp(steering_eq, t_span, init_guess, rtol=tol, atol=tol)
x1, y1 = sol1.y[0], sol1.y[1]











