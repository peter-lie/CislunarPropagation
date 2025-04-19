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
# from mpl_toolkits.mplot3d import Axes3D


mu = 0.012150585609624  # Earth-Moon system mass ratio
omega_S = 1  # Sun's angular velocity (rev/TU)
mass_S = 1.988416e30 / (5.974e24 + 73.48e21) # Sun's mass ratio relative to the Earth-Moon system
dist_S = 149.6e6 / 384.4e3 # Distance of the sun in Earth-moon distances to EM Barycenter
tol = 1e-12 # Tolerancing for accuracy
Omega0 = 0 # RAAN of sun in EM system (align to vernal equinox)
theta0 = 0 # true anomaly of sun at start
inc = 5.145 * (np.pi/180) # Inclination of moon's orbit (sun's ecliptic with respect to the moon)


# BCR4BP Equations of Motion
def bcr4bp_equations(t, state, mu, inc, Omega, theta0):
    # Unpack the state vector
    x, y, z, vx, vy, vz = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

    # Accelerations from the Sun's gravity (transformed)
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t, inc, Omega, theta0)

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + a_Sx
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3 + a_Sy
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3 + a_Sz  

    return [vx, vy, vz, ax, ay, az]

# Sun's position as a function of time (circular motion)
def sun_position(t, inc, Omega0, theta0):
    # Sun's position in the equatorial plane (circular motion)
    r_Sx = dist_S * (np.cos((t+theta0) - Omega0) * np.cos(Omega0) - np.sin((t+theta0) - Omega0) * np.sin(Omega0) * np.cos(inc))
    r_Sy = dist_S * (np.cos((t+theta0) - Omega0) * np.sin(Omega0) + np.sin(- (t+theta0) - Omega0) * np.cos(Omega0) * np.cos(inc))
    r_Sz = dist_S * (np.sin((t+theta0) - Omega0) * np.sin(inc))
    # r_S= np.array([r_Sx_eq, r_Sy_eq, r_Sz_eq])
    # return r_S[0], r_S[1], r_S[2]
    return r_Sx, r_Sy, r_Sz

r_Sx0, r_Sy0, r_Sz0 = sun_position(0, inc, Omega0, theta0)
# r_Sx1, r_Sy1, r_Sz1 = sun_position(0, inc, Omega0, theta0+np.pi/4)
r_Sx1, r_Sy1, r_Sz1 = sun_position(np.pi/6, inc, Omega0, theta0)
r_Sx2, r_Sy2, r_Sz2 = sun_position(np.pi/3, inc, Omega0, theta0)
r_Sx3, r_Sy3, r_Sz3 = sun_position(np.pi/2, inc, Omega0, theta0)
r_Sx4, r_Sy4, r_Sz4 = sun_position(2*np.pi/3, inc, Omega0, theta0)
r_Sx5, r_Sy5, r_Sz5 = sun_position(5*np.pi/6, inc, Omega0, theta0)
r_Sx6, r_Sy6, r_Sz6 = sun_position(np.pi, inc, Omega0, theta0)
r_Sx7, r_Sy7, r_Sz7 = sun_position(7*np.pi/6, inc, Omega0, theta0)
r_Sx8, r_Sy8, r_Sz8 = sun_position(4*np.pi/3, inc, Omega0, theta0)
r_Sx9, r_Sy9, r_Sz9 = sun_position(3*np.pi/2, inc, Omega0, theta0)
r_Sx10, r_Sy10, r_Sz10 = sun_position(5*np.pi/3, inc, Omega0, theta0)
r_Sx11, r_Sy11, r_Sz11 = sun_position(11*np.pi/6, inc, Omega0, theta0)

# Solar Acceleration
def sun_acceleration(x, y, z, t, inc, Omega, theta0):
    # Get Sun's transformed position
    r_Sx, r_Sy, r_Sz = sun_position(t, inc, Omega, theta0)
    
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

# Effective potential U(x, y)
def U(x, y, mu):
    r1, r2 = r1_r2(x, y, mu)
    return 0.5 * (x**2 + y**2) + (1 - mu)/r1 + mu/r2

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
x0, y0, z0 = 1.15, 0, 0  # Initial position
vx0, vy0, vz0 = 0, 0.0086882909, 0      # Initial velocity

state0 = [x0, y0, z0, vx0, vy0, vz0]  # Initial state vector

# Time span for the propagation
t_span = (0, 29.46)  # Start and end times
t_eval = np.linspace(0, 29.46, 1000)  # Times to evaluate the solution

# Solve the system of equations
sol = solve_ivp(bcr4bp_equations, t_span, state0, args=(mu,inc,Omega0,theta0), t_eval=t_eval, rtol=tol, atol=tol)


# Plot Figure
plt.figure(figsize=(10,6))
plt.plot(sol.y[0], sol.y[1], color = 'navy',label='Trajectory')

# Plot Earth and Moon
plt.scatter(-mu, 0, color='blue', s=60, label='Earth')  # Earth at (-mu, 0)
plt.scatter(1 + mu, 0, color='gray', s=15, label='Moon')  # Moon at (1 - mu, 0) 

# # Plot Sun
# plt.scatter(r_Sx0 /200 , r_Sy0 /200, color='yellow', s=80, label='Sun') # Sun at starting position
# plt.scatter(r_Sx1 /200 , r_Sy1 /200, color='yellow', s=80)
# plt.scatter(r_Sx2 /200 , r_Sy2 /200, color='yellow', s=80)
# plt.scatter(r_Sx3 /200 , r_Sy3 /200, color='yellow', s=80)
# plt.text(r_Sx0 /200 -.4 , r_Sy0 /200 - .2, 'Sun @ t = 0')
# plt.text(r_Sx1 /200 -.4 , r_Sy1 /200 + .2, 'Sun @ t = pi/2 TU')
# plt.text(r_Sx2 /200 -.15 , r_Sy2 /200 - .2, 'Sun @ t = pi TU')
# plt.text(r_Sx3 /200 -.4 , r_Sy3 /200 - .2, 'Sun @ t = 3pi/2 TU')

# Plot the Lagrange points
plt.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], color='red', s=15, label='Langrage Points')

plt.xlabel('x [DU]')
plt.ylabel('y [DU]')
# plt.title('CR3BP: Free Return Trajectory')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
# plt.legend()

plt.show()


# 3D Plotting

# # Plot the trajectory
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# # Plot the celestial bodies
# ax.scatter(-mu, 0, 0, color='blue', label='Earth', s=100)  # Primary body (Earth)
# ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=20)  # Secondary body (Moon)

# # Plot the Lagrange points
# # ax.scatter([L2_x], [0], [0], color='red', s=15, label='L2')
# # ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=15, label='Langrage Points')

# # Plot the sun

# ax.scatter(r_Sx0 /100 , r_Sy0 /100, r_Sz0 /100, color=(1,.65,0), s=60, label='Sun') # Sun at starting position
# ax.scatter(r_Sx1 /100 , r_Sy1 /100, r_Sz1 /100, color=(1,.65,0), s=60)
# ax.scatter(r_Sx2 /100 , r_Sy2 /100, r_Sz2 /100, color=(1,.65,0), s=60)
# ax.scatter(r_Sx3 /100 , r_Sy3 /100, r_Sz3 /100, color=(1,.65,0), s=60)
# ax.scatter(r_Sx4 /100 , r_Sy4 /100, r_Sz4 /100, color=(1,.65,0), s=60)
# ax.scatter(r_Sx5 /100 , r_Sy5 /100, r_Sz5 /100, color=(1,.65,0), s=60)
# ax.scatter(r_Sx6 /100 , r_Sy6 /100, r_Sz6 /100, color=(1,.65,0), s=60)
# ax.scatter(r_Sx7 /100 , r_Sy7 /100, r_Sz7 /100, color=(1,.65,0), s=60, alpha=0.65)
# ax.scatter(r_Sx8 /100 , r_Sy8 /100, r_Sz8 /100, color=(1,.65,0), s=60, alpha=0.65)
# ax.scatter(r_Sx9 /100 , r_Sy9 /100, r_Sz9 /100, color=(1,.65,0), s=60, alpha=0.65)
# ax.scatter(r_Sx10 /100 , r_Sy10 /100, r_Sz10 /100, color=(1,.65,0), s=60, alpha=0.65)
# ax.scatter(r_Sx11 /100 , r_Sy11 /100, r_Sz11 /100, color=(1,.65,0), s=60, alpha=0.65)
# # ax.text(r_Sx0 /100 -.4 , r_Sy0 /100 - .2, r_Sz0 /100, 'Sun @ t = 0')
# # ax.text(r_Sx1 /100 -.4 , r_Sy1 /100 + .2, r_Sz1 /100, 'Sun @ t = pi/2 TU')
# # ax.text(r_Sx4 /100 -.15 , r_Sy4 /100 - .2, r_Sz4 /100, 'Sun @ t = pi TU')
# # ax.text(r_Sx3 /100 -.4 , r_Sy3 /100 - .2, r_Sz3 /100, 'Sun @ t = 3pi/2 TU')

# ax.quiver((r_Sx0 + .25*(r_Sx1-r_Sx0))/100, (r_Sy0 + .25*(r_Sy1-r_Sy0))/100, (r_Sz0 + .25*(r_Sz1-r_Sz0))/100, (r_Sx1-r_Sx0)/100 , (r_Sy1-r_Sy0)/100, (r_Sz1-r_Sz0)/100, length = .45, color='black')

# # Labels and plot settings
# ax.set_xlabel('x [DU]')
# ax.set_ylabel('y [DU]')
# # ax.set_axis_off()  # Turn off the axes for better visual appeal
# ax.set_zticks([])
# ax.legend()

# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()



