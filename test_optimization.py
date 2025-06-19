"""
test_optimization
Test cases for optimization problems
AERO 557 Winter 2024
Homework 3 Problem 3
Homework 4 Problem 2

"""


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# from src.cpslo_orbits import optimization as optim
import optimization as optim


"""
Homework 3 Problem 3
Optimize 2D trajectory for maximum horizontal velocity at a 
final height of 1, vertical velocity of 0, starting from rest at origin

"""

# Constants
t_span = [0, 3]  # Transfer time
tol = 1e-9       # Tolerance for ODE and optimization

# Initial guesses
init_guess = [0, 0, 0, 0, 0, -1, -1, -1]

# First propagation with initial guess
sol1 = solve_ivp(optim.steering_eq, t_span, init_guess, rtol=tol, atol=tol)
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
# Desired constraints must be changed inside of local optimization.py file
optimized_init = fsolve(optim.steering_eq_fsolve, init_guess, t_span)

# Propagate with optimized initial conditions
sol2 = solve_ivp(optim.steering_eq, t_span, optimized_init, rtol=tol, atol=tol)
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



"""
Homework 4 Problem 2
Optimize 2D transfer trajectory between circular orbits for lowest delta-V (highest 
ending mass) in polar coordinates

"""


# Constants
Ve = 0.9  # Exhaust velocity
T = 0.1   # Maximum thrust

# Initial conditions
x0, y0 = 1.05, 0
vx0, vy0 = 0, 1.05 ** (-0.5)
m0 = 1
tspan = [0, 6]  # Initial guess for the time span
tol = 1.5e-10       # Tolerance for ODE and optimization

init_state = [x0, y0, vx0, vy0, m0, -1, 0, 0, -1, -1]  # Initial state with adjoint variables

# Solve initial guess trajectory
sol = solve_ivp(optim.burn_coast_burn, tspan, init_state, args=(T, Ve), rtol=tol, atol=tol)

# Extract results
t = sol.t
x, y, vx, vy, m, λ1, λ2, λ3, λ4, λ5 = sol.y

switching_function = np.sqrt(λ3**2 + λ4**2) * Ve + λ5 * m

# Plot trajectory
theta = np.linspace(0, 2 * np.pi, 100)
plt.figure()
plt.plot(x, y, label='Continuous Transfer Orbit')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.grid()
plt.title('Initial Propagation')
plt.show()

# Plot switching function
plt.figure()
plt.plot(t, switching_function, label='Switching Function')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Switching Function')
plt.grid()
plt.title('Switching Function')
plt.show()


# Use Fsolve to get to final position with continuous burn
final_time_guess = 6

fsolve_init_state = np.hstack([init_state, final_time_guess])


optimized_init_burn = fsolve(optim.burn_fsolve, fsolve_init_state)

# Pull out initial guess conditions and final time

optimized_init = optimized_init_burn[0:10] # Q2.S0 = Q2.X(1:10)
tspan1 = optimized_init_burn[10] # Q2.tspan = [0 Q2.X(11)]

t_span = [0, tspan1]

# Propagate with optimized initial conditions
sol2 = solve_ivp(optim.burn_coast_burn, t_span, optimized_init, args=(T, Ve), rtol=tol, atol=tol)

# Extract results
t = sol2.t
x, y, vx, vy, m, λ1, λ2, λ3, λ4, λ5 = sol2.y

switching_function = np.sqrt(λ3**2 + λ4**2) * Ve + λ5 * m

# Plot trajectory
theta = np.linspace(0, 2 * np.pi, 100)
plt.figure()
plt.plot(x, y, label='Continuous Transfer Orbit')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Optimized Propagation')
plt.show()

# Plot switching function
plt.figure()
plt.plot(t, switching_function, label='Switching Function')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Switching Function')
plt.grid()
plt.title('Switching Function')
plt.show()


# guess = [2.6, 4, 6]  # Initial time guesses
# optimal_times = fsolve(burn_fsolve, guess)

tspan3 = [0, 2]
tspan4 = [2, t[-1]]
tspan5 = [t[-1], 1.2*t[-1]]

# Burn
sol3 = solve_ivp(optim.burn_coast_burn, tspan3, optimized_init, args=(T, Ve), rtol=tol, atol=tol)
new_initial = sol3.y[:,-1]

# Coast
sol4 = solve_ivp(optim.burn_coast_burn, tspan4, new_initial, args=(0, Ve), rtol=tol, atol=tol)
new_initial = sol4.y[:,-1]

# Burn
sol5 = solve_ivp(optim.burn_coast_burn, tspan5, new_initial, args=(T, Ve), rtol=tol, atol=tol)

# Combine results
t_combined = np.hstack((sol3.t, sol4.t, sol5.t))
state_combined = np.hstack((sol3.y, sol4.y, sol5.y))

x,y,vx,vy,m,λ1,λ2,λ3,λ4,λ5= state_combined[0,:],state_combined[1,:],state_combined[2,:],state_combined[3,:],state_combined[4,:],state_combined[5,:],state_combined[6,:],state_combined[7,:],state_combined[8,:],state_combined[9,:], 

switching_function = np.sqrt(λ3**2 + λ4**2) * Ve + λ5 * m

# Plot trajectory
theta = np.linspace(0, 2 * np.pi, 100)
plt.figure()
plt.plot(x, y, label='Burn Coast Burn Transfer Orbit')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Initial Propagation')
plt.show()

# Plot switching function
plt.figure()
plt.plot(t_combined, switching_function, label='Switching Function')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Switching Function')
plt.grid()
plt.title('Switching Function')
plt.show()

# Burn Coast Burn Fsolve
# BCB_fsolve_init = np.hstack([optimized_init, 2, t[-1], 1.2*t[-1], 0, 0])
BCB_fsolve_init = np.hstack([optimized_init, 1, 3.6, 1, 0, 0])



# np.scipy.optimize.root
optimized_init_BCB = fsolve(optim.BCB_fsolve, BCB_fsolve_init)

print("optimized_init_BCB:" , optimized_init_BCB)
print("optimized_init_BCB09:" , optimized_init_BCB[0:10])
print("optimized_init_BCB10:" , optimized_init_BCB[10])
print("optimized_init_BCB11:" , optimized_init_BCB[11])
print("optimized_init_BCB12:" , optimized_init_BCB[12])

BCB_optimized_state = optimized_init_BCB[0:10]
optimized_t1 = optimized_init_BCB[10]
optimized_t2 = np.sum(optimized_init_BCB[11] + optimized_init_BCB[10])
optimized_tf = np.sum(optimized_init_BCB[12] + optimized_init_BCB[11] + optimized_init_BCB[10])

tspan3 = [0, optimized_t1]
tspan4 = [optimized_t1, optimized_t2]
tspan5 = [optimized_t2, optimized_tf]

# Burn
sol3 = solve_ivp(optim.burn_coast_burn, tspan3, BCB_optimized_state, args=(T, Ve), rtol=tol, atol=tol)
x3, y3, vx3, vy3, m3, λ1_3, λ2_3, λ3_3, λ4_3, λ5_3 = sol3.y
new_initial = sol3.y[:,-1]

# Coast
sol4 = solve_ivp(optim.burn_coast_burn, tspan4, new_initial, args=(0, Ve), rtol=tol, atol=tol)
x4, y4, vx4, vy4, m4, λ1_4, λ2_4, λ3_4, λ4_4, λ5_4 = sol4.y
new_initial = sol4.y[:,-1]

# Burn
sol5 = solve_ivp(optim.burn_coast_burn, tspan5, new_initial, args=(T, Ve), rtol=tol, atol=tol)
x5, y5, vx5, vy5, m5, λ1_5, λ2_5, λ3_5, λ4_5, λ5_5 = sol5.y

# Combine results
t_combined = np.hstack((sol3.t, sol4.t, sol5.t))
state_combined = np.hstack((sol3.y, sol4.y, sol5.y))

x,y,vx,vy,m,λ1,λ2,λ3,λ4,λ5= state_combined[0,:],state_combined[1,:],state_combined[2,:],state_combined[3,:],state_combined[4,:],state_combined[5,:],state_combined[6,:],state_combined[7,:],state_combined[8,:],state_combined[9,:], 

switching_function = (λ3**2 + λ4**2)**.5 * Ve + λ5 * m

switching_function1 = (λ3_3**2 + λ4_3**2)**.5 * Ve + λ5_3 * m3
switching_function2 = (λ3_4**2 + λ4_4**2)**.5 * Ve + λ5_4 * m4
switching_function3 = (λ3_5**2 + λ4_5**2)**.5 * Ve + λ5_5 * m5



# Plot trajectory
theta = np.linspace(0, 2 * np.pi, 100)
plt.figure()
plt.plot(x, y, label='Burn Coast Burn Transfer Orbit')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Optimized Propagation')
plt.show()


# Trajectory Broken Up
plt.figure()
plt.plot(x3, y3, label='Burn 1', color = 'red')
plt.plot(x4, y4, label='Coast', color = 'green')
plt.plot(x5, y5, label='Burn 2', color = 'purple')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Optimized Propagation')
plt.show()



# Plot switching function
plt.figure()
plt.plot(t_combined, switching_function, label='Switching Function')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Switching Function')
plt.grid()
plt.title('Switching Function')
plt.show()


# Plot switching function
plt.figure()
plt.plot(sol3.t, switching_function1, label='Burn 1', color = 'red')
plt.plot(sol4.t, switching_function2, label='Coast', color = 'green')
plt.plot(sol5.t, switching_function3, label='Burn 2', color = 'purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Switching Function')
plt.show()


# Mass History Plot
plt.figure()
plt.plot(t_combined, m, label='Mass History')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Mass')
plt.grid()
plt.title('Mass History')
plt.show()

