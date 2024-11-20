# Peter Lie
# AERO 599
# Bicircular Restricted 4 Body Problem
# Earth moon system with solar perturbation

import os
clear = lambda: os.system('clear')
clear()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mu = 0.012150585609624  # Earth-Moon system mass ratio
omega_S = 2*np.pi  # Sun's angular velocity (rad/TU)
mass_S = 1.988416e30 / (5.974e24 + 73.48e21) # Sun's mass ratio relative to the Earth-Moon system
dist_S = 149.6e6 / 384.4e3 # Distance of the sun in Earth-moon distances to EM Barycenter


# BCR4BP Equations of Motion
def bcr4bp_equations(t, state, mu):
    # Unpack the state vector
    x, y, z, vx, vy, vz = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

    # Accelerations from the Sun's gravity
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t)

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3 + a_Sx
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3 + a_Sy
    az = -(1 - mu)*z/r1**3 - mu*z/r2**3 + a_Sz  
    
    return [vx, vy, vz, ax, ay, az]

# Sun's position as a function of time (circular motion)
def sun_position(t):
    r_Sx = dist_S * np.cos(-omega_S * t)
    r_Sy = dist_S * np.sin(-omega_S * t)
    r_Sz = 0
    return r_Sx, r_Sy, r_Sz


r_Sx0, r_Sy0, r_Sz0 = sun_position(0)
r_Sx1, r_Sy1, r_Sz1 = sun_position(.25)
r_Sx2, r_Sy2, r_Sz2 = sun_position(.5)
r_Sx3, r_Sy3, r_Sz3 = sun_position(.75)


# Solar Acceleration
def sun_acceleration(x, y, z, t):
    r_Sx, r_Sy, r_Sz = sun_position(t)
    r_S = np.sqrt((x - r_Sx)**2 + (y - r_Sy)**2 + (z - r_Sz)**2)
    dist_S = np.sqrt((r_Sx)**2 + (r_Sy)**2 + (r_Sz)**2)
    a_Sx = -mass_S * (x - r_Sx) / r_S**3 - (mass_S * r_Sx) / dist_S**3
    a_Sy = -mass_S * (y - r_Sy) / r_S**3 - (mass_S * r_Sy) / dist_S**3
    a_Sz = -mass_S * (z - r_Sz) / r_S**3 - (mass_S * r_Sz) / dist_S**3
    return a_Sx, a_Sy, a_Sz

# Distance from satellite to Earth and Moon
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


# Initial Conditions
# RHS
# x0, y0, z0 = -1.15, 0, 0  # Initial position
# vx0, vy0, vz0 = 0, -0.0086882909, 0      # Initial velocity
# x0, y0, z0 = 1.15, 0, 0  # Initial position
# vx0, vy0, vz0 = 0, 0.0086882909, 0      # Initial velocity

# 9:2 L2 Southern NRHO
x0, y0, z0, vx0, vy0, vz0 = [1.0213448959167291E+0,	-4.6715051049863432E-27,	-1.8162633785360355E-1,	-2.3333471915735886E-13,	-1.0177771593237860E-1,	-3.4990116102675334E-12]

state0 = [x0, y0, z0, vx0, vy0, vz0]  # Initial state vector

# Time span for the propagation
t_span = (0, 14)  # Start and end times
t_eval = np.linspace(0, 14, 1000)  # Times to evaluate the solution

# Solve the system of equations
sol = solve_ivp(bcr4bp_equations, t_span, state0, args=(mu,), t_eval=t_eval, rtol=1e-9, atol=1e-9)


# Plot Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Earth and Moon
ax.scatter(-mu, 0, 0, color='blue', label='Earth', s=60)  # Primary body (Earth)
ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=15)  # Secondary body (Moon)

# Plot Solution
ax.plot(sol.y[0], sol.y[1], sol.y[2], color='navy', label='Trajectory')

# Plot Sun
# ax.scatter(r_Sx0 /200 , r_Sy0 /200, 0, color='yellow', s=80, label='Sun') # Sun at starting position
# ax.scatter(r_Sx1 /200 , r_Sy1 /200, 0, color='yellow', s=80)
# ax.scatter(r_Sx2 /200 , r_Sy2 /200, 0, color='yellow', s=80)
# ax.scatter(r_Sx3 /200 , r_Sy3 /200, 0, color='yellow', s=80)
# ax.text(r_Sx0 /200 -.4 , r_Sy0 /200 - .2, 0, 'Sun @ t = 0')
# ax.text(r_Sx1 /200 -.4 , r_Sy1 /200 + .2, 0, 'Sun @ t = .25 TU')
# ax.text(r_Sx2 /200 -.15 , r_Sy2 /200 - .2, 0, 'Sun @ t = .5 TU')
# ax.text(r_Sx3 /200 -.4 , r_Sy3 /200 - .2, 0, 'Sun @ t = .75 TU')

# Plot the Lagrange points
# ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=10, label='Langrage Points')

ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')
# plt.title('CR3BP: Free Return Trajectory')
# ax.grid(True)
ax.legend()
ax.set_box_aspect([1,.4,1]) 

plt.show()

