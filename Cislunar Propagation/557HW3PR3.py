# Peter Lie
# AERO 599
# Orbit optimization in Python


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants
t_span = [0, 3]  # Transfer time
tol = 1e-9       # Tolerance for ODE and optimization

# Define the equations of motion
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

# Fsolve Function
def steering_eq_fsolve(init_state):
    # Propagate the state with the given initial conditions
    sol = solve_ivp(steering_eq, t_span, init_state, rtol=tol, atol=tol)
    final_state = sol.y[:, -1]  # Final state

    # Final constraints
    final_y = final_state[1] - 1  # Final height = 1
    final_ydot = final_state[3]   # Final vertical velocity = 0
    final_lambda1 = final_state[4] # Final lambda1 = 0
    final_lambda3 = final_state[6] + 1 # Final lambda3 = -1
    initial_state1 = init_state[0]  # Initial positions and velocities are 0
    initial_state2 = init_state[1]  # Initial positions and velocities are 0
    initial_state3 = init_state[2]  # Initial positions and velocities are 0
    initial_state4 = init_state[3]  # Initial positions and velocities are 0

    return [final_y, final_ydot, final_lambda1, final_lambda3, initial_state1, initial_state2, initial_state3, initial_state4]

# Initial guesses
init_guess = [0, 0, 0, 0, 0, -1, -1, -1]

# First propagation with initial guess
sol1 = solve_ivp(steering_eq, t_span, init_guess, rtol=tol, atol=tol)
x1, y1 = sol1.y[0], sol1.y[1]

# Plot initial trajectory
plt.figure()
plt.plot(x1, y1, linewidth=1.4)
plt.grid(True, which='both', linestyle='--', linewidth=1.1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Initial Propagation")
plt.show()

print("Guessed initial state:", init_guess)
print("Guessed final state:", sol1.y[:, -1])

# Solve for optimized initial conditions using fsolve
optimized_init = fsolve(steering_eq_fsolve, init_guess)

# Propagate with optimized initial conditions
sol2 = solve_ivp(steering_eq, t_span, optimized_init, rtol=tol, atol=tol)
x2, y2 = sol2.y[0], sol2.y[1]

# Final state
final_state = sol2.y[:, -1]

# Plot optimized trajectory
plt.figure()
plt.plot(x2, y2, linewidth=1.4)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Final Propagation")
plt.show()

# Steering angle over time
beta = np.arctan(sol2.y[7], sol2.y[6]) * 180 / np.pi  # Steering angle in degrees
plt.figure()
plt.plot(sol2.t, beta, linewidth=1.4)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xlabel("Time Units")
plt.ylabel("Steering Angle [deg]")
plt.title("Steering Angle over Time")
plt.show()

# Velocity over time
plt.figure()
plt.plot(sol2.t, sol2.y[2], label="x Velocity", linewidth=1.4)
plt.plot(sol2.t, sol2.y[3], label="y Velocity", linewidth=1.4)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xlabel("Time Units")
plt.ylabel("Velocity [DU/TU]")
plt.title("Velocity over Time")
plt.legend(loc="best")
plt.show()

# Print results
print("Optimized initial state:", optimized_init)
print("Final state:", final_state)
print("Final Position:", [final_state[0], final_state[1]])
print("Final Velocity:", [final_state[2], final_state[3]])
print("Initial Steering Angle:", beta[0], "degrees")
print("Final Steering Angle:", beta[-1], "degrees")