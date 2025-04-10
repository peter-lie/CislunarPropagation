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

# DRO Initial Conditions
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
t_span = (0, 5.9)  # Start and end times
# t_eval = np.linspace(0, 20, 10000)  # Times to evaluate the solution

# Solve the system of equations
sol0 = solve_ivp(cr3bp_equations, t_span, state0, args=(mu,), rtol=1e-12, atol=1e-12)
sol1 = solve_ivp(cr3bp_equations, t_span, state1, args=(mu,), rtol=1e-12, atol=1e-12)
sol2 = solve_ivp(cr3bp_equations, t_span, state2, args=(mu,), rtol=1e-12, atol=1e-12)
sol3 = solve_ivp(cr3bp_equations, t_span, state3, args=(mu,), rtol=1e-12, atol=1e-12)
sol4 = solve_ivp(cr3bp_equations, t_span, state4, args=(mu,), rtol=1e-12, atol=1e-12)
sol5 = solve_ivp(cr3bp_equations, t_span, state5, args=(mu,), rtol=1e-12, atol=1e-12)
sol6 = solve_ivp(cr3bp_equations, t_span, state6, args=(mu,), rtol=1e-12, atol=1e-12)
sol7 = solve_ivp(cr3bp_equations, t_span, state7, args=(mu,), rtol=1e-12, atol=1e-12)


# 3D Plotting

# Plot the trajectory
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the massive bodies
# Define sphere properties
x0, y0, z0 = 1-mu, 0, 0  # center
r = 0.004526             # radius
cmoon = 'gray'           # color
# Create sphere coordinates
u, v = np.linspace(0, 2 * np.pi, 300), np.linspace(0, np.pi, 300)
u, v = np.meshgrid(u, v)
xmoon = x0 + r * np.cos(u) * np.sin(v)
ymoon = y0 + r * np.sin(u) * np.sin(v)
zmoon = z0 + r * np.cos(v)

# Define sphere properties
x0, y0, z0 = -mu, 0, 0   # center
r = 0.016592             # radius
cearth = 'blue'          # color
# Create sphere coordinates
u, v = np.linspace(0, 2 * np.pi, 300), np.linspace(0, np.pi, 300)
u, v = np.meshgrid(u, v)
xearth = x0 + r * np.cos(u) * np.sin(v)
yearth = y0 + r * np.sin(u) * np.sin(v)
zearth = z0 + r * np.cos(v)

# Plot the sphere
ax.plot_surface(xmoon, ymoon, zmoon, color=cmoon, alpha=0.8, linewidth=0)
ax.plot_surface(xearth, yearth, zearth, color=cearth, alpha=0.8, linewidth=0)

# Plot the Lagrange points
ax.scatter([L1_x, L2_x], [0, 0], [0, 0], color='red', s=3, label='Langrage Points')
# ax.scatter([L1_x, L2_x, L4_x, L5_x], [0, 0, L4_y, L5_y], [0, 0, 0, 0], color='red', s=5, label='Langrage Points')
# ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=15, label='Langrage Points')

# 1: [0, 0.4470, 0.7410]        Blue
# 2: [0.8500, 0.3250, 0.0980]   Red
# 3: [0.9290, 0.6940, 0.1250]   Yellow
# 4: [0.4940, 0.1840, 0.5560]   Purple
# 5: [0.4660, 0.6740, 0.1880]   Green?
# 6: [0.3010, 0.7450, 0.9330]
# 7: [0.6350, 0.0780, 0.1840]


# Plot the trajectory of the small object
ax.plot(sol0.y[0], sol0.y[1], sol0.y[2], color=[0,130/255,128/255], label='Trajectory')
ax.plot(sol1.y[0], sol1.y[1], sol1.y[2], color=[0,130/255,128/255])
ax.plot(sol2.y[0], sol2.y[1], sol2.y[2], color=[0.4940, 0.1840, 0.5560])
ax.plot(sol3.y[0], sol3.y[1], sol3.y[2], color=[0,130/255,128/255])
ax.plot(sol4.y[0], sol4.y[1], sol4.y[2], color=[0,130/255,128/255])
ax.plot(sol5.y[0], sol5.y[1], sol5.y[2], color=[0,130/255,128/255])
ax.plot(sol6.y[0], sol6.y[1], sol6.y[2], color=[0,130/255,128/255])
ax.plot(sol7.y[0], sol7.y[1], sol7.y[2], color=[0,130/255,128/255])

# Labels and plot settings
ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')

# xticks = -1, -.5, 0, .5, 1
# ax.set_xticks(xticks)

zticks = -.1, 0, .1
ax.set_zticks(zticks)


ax.set_title('CR3BP Propagation')
# ax.legend()
ax.set_box_aspect([1,1,.12]) 
plt.show()



# # 2D Plotting

# plt.figure(figsize=(10, 6))
# plt.plot(sol0.y[0], sol0.y[1], color = 'navy',label='Trajectory')
# plt.plot(sol1.y[0], sol1.y[1], color = 'navy')
# plt.plot(sol2.y[0], sol2.y[1], color = 'green') # DRO used by McGuire
# plt.plot(sol3.y[0], sol3.y[1], color = 'navy')
# plt.plot(sol4.y[0], sol4.y[1], color = 'navy')
# plt.plot(sol5.y[0], sol5.y[1], color = 'navy')
# plt.plot(sol6.y[0], sol6.y[1], color = 'navy')
# plt.plot(sol7.y[0], sol7.y[1], color = 'navy')

# # Plot Earth and Moon
# plt.scatter(-mu, 0, color='blue', s=60, label='Earth')  # Earth at (-mu, 0)
# plt.scatter(1 - mu, 0, color='gray', s=15, label='Moon')  # Moon at (1 - mu, 0) 

# # Plot the Lagrange points
# plt.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], color='red', s=15, label='Langrage Points')

# plt.xlabel('x [DU]')
# plt.ylabel('y [DU]')
# plt.title('CR3BP: Distant Retrograde Orbit Family')
# plt.grid(True)
# plt.gca().set_aspect('equal', adjustable='box')
# # plt.legend()

# # Turning the background black
# # fig.patch.set_facecolor('black')  # Figure background
# # ax.set_facecolor('black')  
# # ax.set_axis_off()

# plt.show()

