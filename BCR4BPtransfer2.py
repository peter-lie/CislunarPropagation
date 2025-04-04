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


# MATLAB Default colors
# 1: [0, 0.4470, 0.7410]
# 2: [0.8500, 0.3250, 0.0980]
# 3: [0.9290, 0.6940, 0.1250]
# 4: [0.4940, 0.1840, 0.5560]
# 5: [0.4660, 0.6740, 0.1880]
# 6: [0.3010, 0.7450, 0.9330]
# 7: [0.6350, 0.0780, 0.1840]


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

# Effective potential U(x, y) - Use with jacobi constant
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
state0 = [1.0213448959167291E+0,	-4.6715051049863432E-27,	-1.8162633785360355E-1,	-2.3333471915735886E-13,	-1.0177771593237860E-1,	-3.4990116102675334E-12] # 1.5021912429136250E+0 TU period
# state1 is 70000km DRO
state1 = [8.0591079311650515E-1,	2.1618091280991729E-23,	3.4136631163268282E-25,	-8.1806482539864240E-13,	5.1916995982435687E-1,	-5.7262098359472236E-25] # 3.2014543457713667E+0 TU period

time1 = 3.2014543457713667E+0 # TU


# moon distance in km
moondist = (1 - mu - state1[0]) * 384.4e3
# print(moondist): 69937.2 km

# Time span for the propagation 
t_span1 = (0, time1)  # Start and end times
t_span2 = (0, 1*2*np.pi) #
# t_eval = np.linspace(0, 29.46, 1000)  # Times to evaluate the solution, use for ECI plotting


# Solve the IVP
sol0_3BPNRHO = solve_ivp(cr3bp_equations, t_span1, state0, args=(mu,), rtol=tol, atol=tol)
sol0_3BPDRO = solve_ivp(cr3bp_equations, t_span1, state1, args=(mu,), rtol=tol, atol=tol)

# sol1_4BPNRHO = solve_ivp(bcr4bp_equations, t_span2, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
# sol1_4BPDRO = solve_ivp(bcr4bp_equations, t_span2, state1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)


# Hypothetical transfer maneuvers
# Starting with 3BP NRHO characteristics, looking for 3BP DRO characteristics

tspant1 = (0,5.6743149)
tspant2 = (tspant1[1],tspant1[1] + 1.0085)
tspant3 = (tspant2[1],tspant2[1] + 5)
# tspant4 = (tspant3[1],tspant3[1] + 2*time1)


solT0 = solve_ivp(bcr4bp_equations, tspant1, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
newstate2 = solT0.y[:,-1] + [0, 0, 0, -.022, -.2, -solT0.y[5,-1]]
# print(newstate2[2])
solT1 = solve_ivp(bcr4bp_equations, tspant2, newstate2, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)

deltav1 = np.sqrt(.022**2 + .2**2 +  (-solT0.y[5,-1])**2)


# DRO Epoch for targeting
tspanfind = (0,2.17725)
sol0_DROfind = solve_ivp(cr3bp_equations, tspanfind, state1, args=(mu,), rtol=tol, atol=tol)
state1out = sol0_DROfind.y
# xend = 1.0635

# check1 = state1out[0,-1]
# check2 = newstate3[0]
# print(check1)
# print(check2)
# print(np.sqrt(check1[0]**2 + check1[1]**2 + check1[2]**2))

# Wait for this to get to DRO orbit
newstate3 = solT1.y[:,-1] + [0, 0, 0, -solT1.y[3,-1] + state1out[3,-1], -solT1.y[4,-1] + state1out[4,-1], -solT1.y[5,-1]]
# print(newstate3)
# solT2 = solve_ivp(cr3bp_equations, tspant3, newstate3, args=(mu,), rtol=tol, atol=tol) # Used to check error
solT2 = solve_ivp(bcr4bp_equations, tspant3, newstate3, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)

deltav2 = np.sqrt((-solT1.y[3,-1] + state1out[3,-1])**2 + (-solT1.y[4,-1] + state1out[4,-1])**2 + (-solT1.y[5,-1])**2)



deltav = deltav1 + deltav2
# print('deltav: ', deltav, 'DU/TU')
DUtokm = 384.4e3 # kms in 1 DU
TUtoS = 375190.25852 # s in 1 3BP TU
TUtoS4 = 406074.761647 # s in 1 4BP TU
deltavS = deltav * DUtokm / TUtoS4
print('deltavS: ', deltavS, 'km/s')


# 3D Plotting

# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot the celestial bodies
# ax.scatter(-mu, 0, 0, color='blue', label='Earth', s=100)  # Primary body (Earth)
ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=30)  # Secondary body (Moon)

# Plot the Lagrange points
ax.scatter([L1_x], [0], [0], color=[0.4660, 0.6740, 0.1880], s=15, label='L1')
ax.scatter([L2_x], [0], [0], color=[0.3010, 0.7450, 0.9330], s=15, label='L2')

# ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=15, label='Langrage Points')

# Plot the trajectories
ax.plot(sol0_3BPNRHO.y[0], sol0_3BPNRHO.y[1], sol0_3BPNRHO.y[2], color=[0, 0.4470, 0.7410], label='9:2 NRHO')
ax.plot(sol0_3BPDRO.y[0], sol0_3BPDRO.y[1], sol0_3BPDRO.y[2], color=[0.4940, 0.1840, 0.5560], label='70000km DRO')
# ax.plot(sol0_DROfind.y[0], sol0_DROfind.y[1], sol0_DROfind.y[2], color=[0.4940, 0.1840, 0.5560], label='Target DRO')

ax.plot(solT0.y[0], solT0.y[1], solT0.y[2], color=[0.9290, 0.6940, 0.1250], label='T 1')
ax.scatter([newstate2[0]], [newstate2[1]], [newstate2[2]], color=[0.8500, 0.3250, 0.0980], s=10, label='Maneuver')
ax.plot(solT1.y[0], solT1.y[1], solT1.y[2], color=[0.4660, 0.6740, 0.1880], label='T 2')

ax.scatter([newstate3[0]], [newstate3[1]], [newstate3[2]], color=[0.8500, 0.3250, 0.0980], s=10)
# ax.plot(solT2.y[0], solT2.y[1], solT2.y[2], color=[0, 0.4470, 0.7410], label='T 3')

#ax.scatter([newstate4[0]], [newstate4[1]], [newstate4[2]], color=[0.8500, 0.3250, 0.0980], s=10)
# ax.plot(solT3.y[0], solT3.y[1], solT3.y[2], color=[0.4660, 0.6740, 0.1880], label='T 4')

# ax.plot(sol1_4BPNRHO.y[0], sol1_4BPNRHO.y[1], sol1_4BPNRHO.y[2], color=[0.9290, 0.6940, 0.1250], label='9:2 NRHO')
# ax.plot(sol1_4BPDRO.y[0], sol1_4BPDRO.y[1], sol1_4BPDRO.y[2], color=[0.4940, 0.1840, 0.5560], label='70000km DRO')


# Labels and plot settings
ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')

# ax.set_axis_off()  # Turn off the axes for better visual appeal

ax.legend(loc='best')

# plt.gca().set_aspect('equal', adjustable='box')
plt.show()



