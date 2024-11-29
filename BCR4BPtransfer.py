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


# BCR4BP Equations of Motion
def bcr4bp_equations(t, state, mu, inc, Omega):
    # Unpack the state vector
    x, y, z, vx, vy, vz = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

    # Accelerations from the Sun's gravity (transformed)
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t, inc, Omega)

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + a_Sx
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3 + a_Sy
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3 + a_Sz  

    return [vx, vy, vz, ax, ay, az]


# Sun's position as a function of time (circular motion)
def sun_position(t, inc, Omega0):
    # Sun's position in the equatorial plane (circular motion)
    r_Sx = dist_S * (np.cos((t+theta0) - Omega0) * np.cos(Omega0) - np.sin((t+theta0) - Omega0) * np.sin(Omega0) * np.cos(inc))
    r_Sy = dist_S * (np.cos((t+theta0) - Omega0) * np.sin(Omega0) + np.sin(- (t+theta0) - Omega0) * np.cos(Omega0) * np.cos(inc))
    r_Sz = dist_S * (np.sin((t+theta0) - Omega0) * np.sin(inc))
    # r_S= np.array([r_Sx_eq, r_Sy_eq, r_Sz_eq])
    # return r_S[0], r_S[1], r_S[2]
    return r_Sx, r_Sy, r_Sz


r_Sx0, r_Sy0, r_Sz0 = sun_position(0, inc, Omega0)
r_Sx1, r_Sy1, r_Sz1 = sun_position(np.pi/6, inc, Omega0)
r_Sx2, r_Sy2, r_Sz2 = sun_position(np.pi/3, inc, Omega0)
r_Sx3, r_Sy3, r_Sz3 = sun_position(np.pi/2, inc, Omega0)
r_Sx4, r_Sy4, r_Sz4 = sun_position(2*np.pi/3, inc, Omega0)
r_Sx5, r_Sy5, r_Sz5 = sun_position(5*np.pi/6, inc, Omega0)
r_Sx6, r_Sy6, r_Sz6 = sun_position(np.pi, inc, Omega0)
r_Sx7, r_Sy7, r_Sz7 = sun_position(7*np.pi/6, inc, Omega0)
r_Sx8, r_Sy8, r_Sz8 = sun_position(4*np.pi/3, inc, Omega0)
r_Sx9, r_Sy9, r_Sz9 = sun_position(3*np.pi/2, inc, Omega0)
r_Sx10, r_Sy10, r_Sz10 = sun_position(5*np.pi/3, inc, Omega0)
r_Sx11, r_Sy11, r_Sz11 = sun_position(11*np.pi/6, inc, Omega0)


def sun_acceleration(x, y, z, t, inc, Omega):
    # Get Sun's transformed position
    r_Sx, r_Sy, r_Sz = sun_position(t, inc, Omega)
    
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
# RHS FRT
# x0, y0, z0 = -1.15, 0, 0  # Initial position
# vx0, vy0, vz0 = 0, -0.0086882909, 0      # Initial velocity

# state0 is 9:2 NRHO
state0 = [1.0213448959167291E+0,	-4.6715051049863432E-27,	-1.8162633785360355E-1,	-2.3333471915735886E-13,	-1.0177771593237860E-1,	-3.4990116102675334E-12]
# state1 is 70000km DRO
state1 = [8.0591079311650515E-1,	2.1618091280991729E-23,	3.4136631163268282E-25,	-8.1806482539864240E-13,	5.1916995982435687E-1,	-5.7262098359472236E-25]

# moon distance in km
moondist = (1 - mu - state1[0]) * 384.4e3
print(moondist)

# Time span for the propagation
t_span = (0, 3.5)  # Start and end times
# t_eval = np.linspace(0, 29.46, 1000)  # Times to evaluate the solution

# Solve the system of equations
sol0 = solve_ivp(cr3bp_equations, t_span, state0, args=(mu,), rtol=tol, atol=tol)
sol1 = solve_ivp(cr3bp_equations, t_span, state1, args=(mu,), rtol=tol, atol=tol)

sol2 = solve_ivp(bcr4bp_equations, t_span, state0, args=(mu,inc,Omega0), rtol=tol, atol=tol)
sol3 = solve_ivp(bcr4bp_equations, t_span, state1, args=(mu,inc,Omega0), rtol=tol, atol=tol)



# 3D Plotting

# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot the celestial bodies
# ax.scatter(-mu, 0, 0, color='blue', label='Earth', s=100)  # Primary body (Earth)
ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=25)  # Secondary body (Moon)

# Plot the Lagrange points
ax.scatter([L1_x], [0], [0], color='red', s=15, label='L1')
ax.scatter([L2_x], [0], [0], color='blue', s=15, label='L2')

# ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=15, label='Langrage Points')

# Plot the trajectories
ax.plot(sol0.y[0], sol0.y[1], sol0.y[2], color='orange', label='9:2 NRHO')
ax.plot(sol1.y[0], sol1.y[1], sol1.y[2], color='green', label='70000km DRO')

# Labels and plot settings
ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')

# ax.set_axis_off()  # Turn off the axes for better visual appeal

ax.legend()

plt.gca().set_aspect('equal', adjustable='box')
plt.show()



