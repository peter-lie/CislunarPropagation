# Peter Lie
# AERO 599
# Python Testing

import os
clear = lambda: os.system('clear')
clear()

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


mu = 398600.4418      # km^3 / s^2

def two_body_equations(t, y):

    # Defines the two-body equations of motion.
    
    # Parameters 
    # t : float, Time variable (not used in this case, but required by solve_ivp)
    # y : array, Array of state variables [x, y, vx, vy].
    
    # Returns:
    # dydt : array, Time derivatives [vx, vy, ax, ay].
    
    x, y, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y**2 + z**2)  # Distance between the two bodies

    # Acceleration due to gravity
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3
    
    return [vx, vy, vz, ax, ay, az]


# Initial conditions: position (x0, y0) and velocity (vx0, vy0)

x0 = 3207       # km (e.g., initial distance from the center, roughly Earth orbit altitude)
y0 = 5459       # km
z0 = 2714       # km
vx0 = -6.532    # km/s
vy0 = 0.7835    # km/s (e.g., orbital speed for a low Earth orbit)
vz0 = 6.142     # km/s

# Combine initial conditions into a single array
initial_conditions = [x0, y0, z0, vx0, vy0, vz0]

# Time span for the simulation (e.g., 10,000 seconds)
t_span = [0, 15000]  # start and end time in seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)
tol = 1e-9 # tolerance

# Solve the ODE using scipy's solve_ivp
solution = solve_ivp(two_body_equations, t_span, initial_conditions, t_eval=t_eval, method='RK45', rtol = tol, atol = tol)

# Extract the results
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]


# Plot the results
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z) 
# ax.set_title('3D Two-Body Orbit Simulation')
# ax.set_xlabel('x (km)')
# ax.set_ylabel('y (km)')
# ax.set_zlabel('z (km)')
# ax.grid(True)

# max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

# Compute midpoints for centering the axes
# mid_x = (x.max() + x.min()) * 0.5
# mid_y = (y.max() + y.min()) * 0.5
# mid_z = (z.max() + z.min()) * 0.5

#set_axes_equal(ax)

#ax.set_xlim(mid_x - max_range, mid_x + max_range)
#ax.set_ylim(mid_x - max_range, mid_x + max_range)
#ax.set_zlim(mid_x - max_range, mid_x + max_range)

# earth_radius = 6378

# u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
# sphere_x = earth_radius * np.cos(u) * np.sin(v)
# sphere_y = earth_radius * np.sin(u) * np.sin(v)
# sphere_z = earth_radius * np.cos(v)

# Plot the sphere representing Earth
# ax.plot_surface(sphere_x, sphere_y, sphere_z, color='g', alpha=0.6, label='Earth', edgecolor = 'w',linewidth = .3)
# fig.patch.set_facecolor('black')  # Figure background
# ax.set_facecolor('black')  
# ax.set_axis_off()
# ax.grid(False)

# plt.show()


# 384400 km
# t∗ = 375190.25852 seconds


# data1 = [1.3632096570, 1.4748399512, 1.5872714606, 1.7008482705, 1.8155211042]
# data2 = [1.0110350588, 1.0192741002, 1.0277926091, 1.0362652156, 1.0445681848]

# plt.figure(figsize=(8,8))
# plt.plot(data1, data2, 'o-')
# plt.plot(1.50206, 1.02134 ,'x')
# plt.xlabel('Period')
# plt.ylabel('x0')
# plt.show()


# Animation for Satellite

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('3D Two-Body Orbit Simulation')
# ax.set_xlabel('x (km)')
# ax.set_ylabel('y (km)')
# ax.set_zlabel('z (km)')
# ax.grid(True)
# ax.plot(x, y, z, label='Trajectory') 
# ax.scatter(0, 0, 0, label="Earth", s=100)
# ax.scatter(x[0], y[0], z[0], label="Sat")

# ax.legend
# plt.show()


# Example trajectory (replace with your actual data)
n = 1000
t = solution.t
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x, y, z, label='Trajectory')

# Plot the Earth at the origin
earth = ax.scatter(0, 0, 0, s=100, label='Earth')

# Initialize the satellite's position
satellite, = ax.plot([x[0]], [y[0]], [z[0]], 'o', label='Satellite')

# Number of frames for the animation
num_frames = 100

# Update function for animation
def animation_frame(frame):
    # Calculate the current index
    index = int(frame * (n / num_frames))
    
    # Update the satellite's position
    # ax.clear
    satellite = ax.scatter([x[index]], [y[index]], [z[index]], color='orange')
    # satellite.set_xdata([x[index]])
    # satellite.set_ydata([y[index]])
    return satellite,

# Create the animation
animation = FuncAnimation(fig, func=animation_frame, frames=num_frames, interval=50)

# Display the animation
plt.legend()
plt.show()
