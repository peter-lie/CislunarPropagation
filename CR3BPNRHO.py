# Peter Lie
# AERO 599
# Circular Restricted 3 Body Problem
# Earth moon system
# Right handed system

import os
clear = lambda: os.system('clear')
clear()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Earth at (-mustar,0)
# Moon at (1-mustar,0)

# CR3BP Equations of Motion
def cr3bp_equations(t, state, mu):
    # Unpack the state vector
    x, y, z, vx, vy, vz = state
    
    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

    # Equations of motion
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    az = -(1 - mu)*z/r1**3 - mu*z/r2**3
    
    return [vx, vy, vz, ax, ay, az]


# Initial conditions and parameters
mu = 0.012150585609624  # Earth-Moon system mass ratio

ER = 6378 / 384400 # Earth Radius in DU

def r1_r2(x, y, z, mu):
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to Earth
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)  # Distance to Moon
    return r1, r2


# Function to find the roots for L1, L2, L3 along the x-axis
def lagrange_x_eq(x, mu, point):
    r1 = abs(x + mu)
    r2 = abs(x - (1 - mu))
    if point == 'L1':
        return x - (1 - mu)/(x + mu)**2 + mu/(x - (1 - mu))**2
    elif point == 'L2':
        return x - (1 - mu)/(x + mu)**2 - mu/(x - (1 - mu))**2
    elif point == 'L3':
        return x + (1 - mu)/(x + mu)**2 + mu/(x - (1 - mu))**2

# Solving for L1, L2, L3 along the x-axis
L1_x = fsolve(lagrange_x_eq, 0.8, args=(mu, 'L1'))[0]
L2_x = fsolve(lagrange_x_eq, 1.2, args=(mu, 'L2'))[0]
L3_x = fsolve(lagrange_x_eq, - 1.2, args=(mu, 'L3'))[0]

# Coordinates for L4 and L5 (equilateral triangle points)
L4_x = 0.5 - mu
L4_y = np.sqrt(3) / 2
L5_x = 0.5 - mu
L5_y = -np.sqrt(3) / 2


# Free Return Initial Conditions

#RHS
# x0, y0, z0 = 1.15, 0, 0  # Initial position
# vx0, vy0, vz0 = 0, 0.0086882909, 0      # Initial velocity


# NRHO Initial Conditions
# x0, y0, z0 = 1.0277926091, 0, -0.1858044184  # Initial position
# vx0, vy0, vz0 = 0, -0.1154896637, 0      # Initial velocity

# 9:2 NRHO Initial Conditions
# Not specific enough??
# x0, y0, z0 = 1.02134, 0, -0.18162  # Initial position
# vx0, vy0, vz0 = 0, -0.10176, 0      # Initial velocity

x0, y0, z0, vx0, vy0, vz0 = [1.0213448959167291E+0,	-4.6715051049863432E-27,	-1.8162633785360355E-1,	-2.3333471915735886E-13,	-1.0177771593237860E-1,	-3.4990116102675334E-12]

state0 = [x0, y0, z0, vx0, vy0, vz0]  # Initial state vector

# Time span for the propagation
t_span = (0, 20)  # Start and end times
# t_eval = np.linspace(0, 20, 10000)  # Times to evaluate the solution

# Solve the system of equations
sol = solve_ivp(cr3bp_equations, t_span, state0, args=(mu,), rtol=1e-12, atol=1e-12)


# 3D Plotting

# Plot the trajectory
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Plot the massive bodies
# ax.scatter(-mu, 0, 0, color='blue', label='Earth', s=80)  # Primary body (Earth)
# ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=20)  # Secondary body (Moon)

# Plot the Lagrange points
# ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=15, label='Langrage Points')

# Plot the trajectory of the small object
# ax.plot(sol.y[0], sol.y[1], sol.y[2], color='navy', label='Trajectory')

# Labels and plot settings
# ax.set_xlabel('x [DU]')
# ax.set_ylabel('y [DU]')
# ax.set_zlabel('z [DU]')
# ax.set_title('CR3BP Propagation')
# ax.legend()
# ax.set_box_aspect([1,.4,1]) 
# plt.show()


# 2D Plotting

# plt.figure(figsize=(8, 8))
# plt.plot(sol.y[0], sol.y[1], color = 'navy',label='Free Return')

# Plot Earth and Moon
# plt.scatter(-mu, 0, color='blue', s=60, label='Earth')  # Earth at (-mu, 0)
# plt.scatter(1 - mu, 0, color='gray', s=15, label='Moon')  # Moon at (1 - mu, 0) 

# Plot the Lagrange points
# plt.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], color='red', s=15, label='Langrage Points')

# plt.xlabel('x [DU]')
# plt.ylabel('y [DU]')
# plt.title('CR3BP: Free Return Trajectory')
# plt.grid(True)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.legend()

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(9, 5))
fig.suptitle("CR3BP: L2S 9:2 NRHO")

# Top-down view (XY plane)
axes[0].plot(sol.y[0], sol.y[1], label="Trajectory")
axes[0].scatter([1 - mu], [0], s=15, color='gray', label="Moon")
axes[0].set_title("Top-down view (XY plane)")
axes[0].set_xlabel("x [DU]")
axes[0].set_ylabel("y [DU]")
axes[0].grid(True)
axes[0].legend()

# Side view (YZ plane)
axes[1].plot(sol.y[1], sol.y[2],label="Trajectory")
axes[1].scatter([0], [0], s=15, color='gray', label="Moon")
axes[1].set_title("Side view (YZ plane)")
axes[1].set_xlabel("y [DU]")
axes[1].set_ylabel("z [DU]")
axes[1].grid(True)
axes[1].legend()

# Moon axis view (XZ plane)
axes[2].plot(sol.y[0], sol.y[2], label="Trajectory")
axes[2].scatter([1 - mu], [0], s=15, color='gray', label="Moon")
axes[2].set_title("Side view (XZ plane)")
axes[2].set_xlabel("x [DU]")
axes[2].set_ylabel("z [DU]")
axes[2].grid(True)
axes[2].legend()

# Adjust layout and show
plt.tight_layout()
plt.show()


# Turning the background black
# fig.patch.set_facecolor('black')  # Figure background
# ax.set_facecolor('black')  
# ax.set_axis_off()

# plt.show()




