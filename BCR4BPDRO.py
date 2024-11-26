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
omega_S = 1  # Sun's angular velocity (rev/TU)
mass_S = 1.988416e30 / (5.974e24 + 73.48e21) # Sun's mass ratio relative to the Earth-Moon system
dist_S = 149.6e6 / 384.4e3 # Distance of the sun in Earth-moon distances to EM Barycenter
tol = 1e-12 # Tolerancing for accuracy
Omega0 = 0 # RAAN of sun in EM system (align to vernal equinox)
inc = 5.145 * (np.pi/180) # Inclination of moon's orbit (sun's ecliptic with respect to the moon)


# BCR4BP Equations of Motion
def bcr4bp_equations(t, state, mu, i, Omega):
    # Unpack the state vector
    x, y, z, vx, vy, vz = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

    # Accelerations from the Sun's gravity (transformed)
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t, i, Omega)

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + a_Sx
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3 + a_Sy
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3 + a_Sz  
    
    # Check magnitudes
    Emag = np.sqrt(ax**2 + ay**2 + az**2)
    Smag = np.sqrt(a_Sx**2 + a_Sy**2 + a_Sz**2)
    ESratio = Emag / Smag

    # print('ESratio: ', ESratio)

    return [vx, vy, vz, ax, ay, az]


def sun_position(t, inc, Omega0):
    # Sun's position in the equatorial plane (circular motion)
    r_Sx = dist_S * (np.cos(t - Omega0) * np.cos(Omega0) - np.sin(t - Omega0) * np.sin(Omega0) * np.cos(inc))
    r_Sy = dist_S * (np.cos(t - Omega0) * np.sin(Omega0) + np.sin(- t - Omega0) * np.cos(Omega0) * np.cos(inc))
    r_Sz = dist_S * (np.sin(t - Omega0) * np.sin(inc))
    # r_S= np.array([r_Sx_eq, r_Sy_eq, r_Sz_eq])
    # return r_S[0], r_S[1], r_S[2]
    return r_Sx, r_Sy, r_Sz

r_Sx0, r_Sy0, r_Sz0 = sun_position(0, inc, Omega0)
r_Sx1, r_Sy1, r_Sz1 = sun_position(np.pi/2, inc, Omega0)
r_Sx2, r_Sy2, r_Sz2 = sun_position(np.pi, inc, Omega0)
r_Sx3, r_Sy3, r_Sz3 = sun_position(3*np.pi/2, inc, Omega0)


# Solar Acceleration
def sun_acceleration(x, y, z, t, i, Omega):
    # Get Sun's transformed position
    r_Sx, r_Sy, r_Sz = sun_position(t, i, Omega)
    
    # Relative distance to the Sun
    r_S = np.sqrt((x - r_Sx)**2 + (y - r_Sy)**2 + (z - r_Sz)**2)
    dist_S = np.sqrt(r_Sx**2 + r_Sy**2 + r_Sz**2)
    
    # Accelerations
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

# Initial State Vectors
# DRO family
x0 = [0.9624690577, 0.9361690577, 0.9098690577, 0.8835690577, 0.8572690577, 
0.8309690577, 0.8046690577, 0.7783690577, 0.7520690577, 0.7257690577, 
0.6993690577, 0.6730690577, 0.6467690577, 0.6204690577, 0.5941690577, 
0.5678690577, 0.5415690577, 0.5152690577, 0.4889690577, 0.4626690577]
vy0 = [0.7184165432, 0.5420829797, 0.4861304073, 0.4704011643, 0.4752314911, 
0.4936194567, 0.5206492176, 0.5556485484, 0.5960858865, 0.6344388745, 
0.6947808121, 0.7502922555, 0.8093649527, 0.8716714431, 0.9376934111, 
1.0072891703, 1.0805472211, 1.1576784834, 1.2390293868, 1.3253326531]

state0 = [x0[2], 0, 0, 0, vy0[2], 0]  # Initial state vector
state1 = [x0[4], 0, 0, 0, vy0[4], 0]  # Initial state vector
state2 = [x0[6], 0, 0, 0, vy0[6], 0]  # Initial state vector
state3 = [x0[8], 0, 0, 0, vy0[8], 0]  # Initial state vector
state4 = [x0[10], 0, 0, 0, vy0[10], 0]  # Initial state vector
state5 = [x0[12], 0, 0, 0, vy0[12], 0]  # Initial state vector
state6 = [x0[14], 0, 0, 0, vy0[14], 0]  # Initial state vector
state7 = [x0[16], 0, 0, 0, vy0[16], 0]  # Initial state vector
state8 = [x0[18], 0, 0, 0, vy0[18], 0]  # Initial state vector
# state9 = [x0[20], 0, 0, 0, vy0[20], 0]  # Initial state vector

# Time span for the propagation
t_span = (0, 6.2)  # Start and end times

# Solve the system of equations
sol0 = solve_ivp(bcr4bp_equations, t_span, state0, args=(mu, inc, Omega0,), rtol=tol, atol=tol)
sol1 = solve_ivp(bcr4bp_equations, t_span, state1, args=(mu, inc, Omega0,), rtol=tol, atol=tol)
sol2 = solve_ivp(bcr4bp_equations, t_span, state2, args=(mu, inc, Omega0,), rtol=tol, atol=tol)
sol3 = solve_ivp(bcr4bp_equations, t_span, state3, args=(mu, inc, Omega0,), rtol=tol, atol=tol)
sol4 = solve_ivp(bcr4bp_equations, t_span, state4, args=(mu, inc, Omega0,), rtol=tol, atol=tol)
sol5 = solve_ivp(bcr4bp_equations, t_span, state5, args=(mu, inc, Omega0,), rtol=tol, atol=tol)
sol6 = solve_ivp(bcr4bp_equations, t_span, state6, args=(mu, inc, Omega0,), rtol=tol, atol=tol)
sol7 = solve_ivp(bcr4bp_equations, t_span, state7, args=(mu, inc, Omega0,), rtol=tol, atol=tol)
sol8 = solve_ivp(bcr4bp_equations, t_span, state8, args=(mu, inc, Omega0,), rtol=tol, atol=tol)




# 2D Plotting

plt.figure(figsize=(8, 8))
plt.plot(sol0.y[0], sol0.y[1], color = 'navy',label='Trajectory')
plt.plot(sol1.y[0], sol1.y[1], color = 'navy')
plt.plot(sol2.y[0], sol2.y[1], color = 'green') # DRO used by McGuire
plt.plot(sol3.y[0], sol3.y[1], color = 'navy')
plt.plot(sol4.y[0], sol4.y[1], color = 'navy')
plt.plot(sol5.y[0], sol5.y[1], color = 'navy')
plt.plot(sol6.y[0], sol6.y[1], color = 'navy')
plt.plot(sol7.y[0], sol7.y[1], color = 'navy')

# Plot Earth and Moon
plt.scatter(-mu, 0, color='blue', s=60, label='Earth')  # Earth at (-mu, 0)
plt.scatter(1 - mu, 0, color='gray', s=15, label='Moon')  # Moon at (1 - mu, 0) 

# Plot the Lagrange points
plt.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], color='red', s=15, label='Langrage Points')

plt.xlabel('x [DU]')
plt.ylabel('y [DU]')
plt.title('CR3BP: Distant Retrograde Orbit Family')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()

# Turning the background black
# fig.patch.set_facecolor('black')  # Figure background
# ax.set_facecolor('black')  
# ax.set_axis_off()

plt.show()

