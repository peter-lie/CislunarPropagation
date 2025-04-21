# Peter Lie
# AERO 599
# Bicircular Restricted 4 Body Problem
# Earth moon system with solar perturbation, varying starting sun true anomaly

import os
clear = lambda: os.system('clear')
clear()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# MATLAB Default colors
# 1: [0, 0.4470, 0.7410]        Blue
# 2: [0.8500, 0.3250, 0.0980]   Red
# 3: [0.9290, 0.6940, 0.1250]   Yellow
# 4: [0.4940, 0.1840, 0.5560]   Purple
# 5: [0.4660, 0.6740, 0.1880]
# 6: [0.3010, 0.7450, 0.9330]
# 7: [0.6350, 0.0780, 0.1840]

# Initialize variables
mu = 0.012150585609624  # Earth-Moon system mass ratio
omega_S = 1  # Sun's angular velocity (rev/TU)
mass_S = 1.988416e30 / (5.974e24 + 73.48e21) # Sun's mass ratio relative to the Earth-Moon system
dist_S = 149.6e6 / 384.4e3 # Distance of the sun in Earth-moon distances to EM Barycenter
tol = 1e-12 # Tolerancing for accuracy
Omega0 = 0 # RAAN of sun in EM system (align to vernal equinox)
theta0 = 13*np.pi/8 # true anomaly of sun at start
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

# Moon centered position
def MCI(x, y, z, mu):
    # moves position from canonical in to moon centered kilometers, for J2 calculation
    xMCI = (x - (1-mu)) * 384400
    yMCI = y * 384400
    zMCI = z * 384400

    return [xMCI, yMCI, zMCI]

# Moon J2 perturbation
def J2(x, y, z, mu):
    # J2 for moon in the moon frame, read in canonical units
    J2val = 202.7e-6
    Rmoon = 1737.5 # km

    # convert from km/s to DU/TU^2
    DUtokm = 384.4e3 # kms in 1 DU
    TUtoS  = 375190.25852 # s in 1 3BP TU

    # position read in as canonical, needs to be in MCI for moon J2
    [xMCI, yMCI, zMCI] = MCI(x, y, z, mu)
    rmoonsc = np.sqrt(xMCI**2 + yMCI**2 + zMCI**2) # now also in km

    # appears to get up to e-13 for NRHO
    aJ2 = [- (3 * J2val * mu * Rmoon**2 / (2 * rmoonsc**5)) * (1 - 5 * (zMCI/rmoonsc)**2) * xMCI, - (3 * J2val * mu * Rmoon**2 / (2 * rmoonsc**5)) * (1 - 5 * (zMCI/rmoonsc)**2) * yMCI, - (3 * J2val * mu * Rmoon**2 / (2 * rmoonsc**5))* (3 - 5 * (zMCI/rmoonsc)**2) * zMCI]

    # convert from km/s^2 to DU/TU^2
    DUtokm = 384.4e3 # kms in 1 DU
    TUtoS  = 375190.25852 # s in 1 3BP TU

    aJ2_canonical = aJ2
    aJ2_canonical[0] = aJ2[0] / DUtokm * TUtoS**2
    aJ2_canonical[1] = aJ2[1] / DUtokm * TUtoS**2
    aJ2_canonical[2] = aJ2[2] / DUtokm * TUtoS**2

    return aJ2_canonical


    # moves position from canonical in to moon centered kilometers, for J2 calculation
    xMCI = (x - (1-mu)) * 384400
    yMCI = y * 384400
    zMCI = z * 384400

    return [xMCI, yMCI, zMCI]

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
# state0 is 9:2 NRHO
state0 = [1.0213448959167291E+0,	-4.6715051049863432E-27,	-1.8162633785360355E-1,	-2.3333471915735886E-13,	-1.0177771593237860E-1,	-3.4990116102675334E-12] # 1.5021912429136250E+0 TU period
# state1 is 70000km DRO
state1 = [8.0591079311650515E-1,	2.1618091280991729E-23,	3.4136631163268282E-25,	-8.1806482539864240E-13,	5.1916995982435687E-1,	-5.7262098359472236E-25] # 3.2014543457713667E+0 TU period

time1 = 3.2014543457713667E+0 # TU


# moon distance in km
moondist = (1 - mu - state1[0]) * 384.4e3
# print(moondist): 69937.2 km


# Check function for DRO distance
from functools import wraps
from typing import List, Union, Optional, Callable

def event_listener():
    """
    Custom decorator to set direction and terminal values for event handling
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.direction = -1
        wrapper.terminal = True
        return wrapper

    return decorator

@event_listener()
def DRO_event(time: float, state: Union[List, np.ndarray], *opts):
    """
    Event listener for `solve_ivp` to quit integration when crossing DRO
    """

   # Compute current position of the spacecraft

    x = state[0]
    y = state[1]

    # Find point closest to in circleplot

    space = np.linspace(0,360)
    circleplotx = np.sqrt(.035) * np.sin(2*np.pi*space/100) + (1-.68*mu)
    circleploty = np.sqrt(.061) * np.cos(2*np.pi*space/100)
    distunder = 1

    for i in range(1,len(space)):

        distance = (x - circleplotx[i])**2 + (y - circleploty[i])**2
        # This can miss and go through if too low
        if distance < .0001:
            # See if greater than that point
    
            distunder = (circleplotx[i]-x) + (circleploty[i]-y)

    # Cross from positive to negative
    # output = moondistSQ - distance
    output = distunder
    # print(output)

    return output


# Time span for the propagation 
t_span1 = (0, time1)  # Start and end times
t_span2 = (0, 1*2*np.pi) #


# Solve the IVP
sol0_3BPNRHO = solve_ivp(cr3bp_equations, t_span1, state0, args=(mu,), rtol=tol, atol=tol)
sol0_3BPDRO = solve_ivp(cr3bp_equations, t_span1, state1, args=(mu,), rtol=tol, atol=tol)

# sol1_4BPNRHO = solve_ivp(bcr4bp_equations, t_span2, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
# sol1_4BPDRO = solve_ivp(bcr4bp_equations, t_span2, state1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)

# Hypothetical transfer maneuvers
# Starting with 3BP NRHO characteristics, looking for 3BP DRO characteristics

# Loop to check for the last time orbit crosses the xy plane inside of the DRO

# theta0 = 0
# theta0 = 197 * np.pi / 128
theta0 = 4.822835597112472
# theta0 = 4.344233591292136
# theta0 = 0.42951462060797985

print('theta0: ', theta0)


deltavmin = 1
thetamin = 0
thrustangle = np.pi/4; # rad, 45 deg .584 with 16 points, .421 with 512 points
# thrustangle = np.pi/3; # rad, 60 deg .584 with 16 points, .460 with 512 points
# thrustangle = np.pi/6; # rad, 30 deg .584 with 16 points, .436 with 512 points
# thrustangle = np.pi/2; # rad, 90 deg .584 with 16 points, .405 with 512 points
# thrustangle = 5*np.pi/12; # rad, 75 deg .584 with 16 points, .415 with 512 points
# thrustangle = np.pi/12; # rad, 15 deg .584 with 16 points, .440 with 512 points
# thrustangle = 0; # rad, 0 deg .584 with 16 points, . with 512 points

# vyoffset = 0
vyoffset = 0    # 0 gives 0.521 km/s with 32 points
                # -.1 yeilds 0.483 km/s with 64 points
                # negative y benefits the points close to the positive x axis, and allows for better curvature

DUtokm = 384.4e3 # kms in 1 DU
TUtoS4 = 406074.761647 # s in 1 4BP TU
vyoffsetKMS = vyoffset * DUtokm / TUtoS4

print('offset is: ', vyoffsetKMS, ' km/s')

moondistSQ = (1*(moondist/384.4e3))**2

# thrustangle = np.pi / 2 # 90 deg

tspant1 = (0,10) # for 0 z position
solT0 = solve_ivp(bcr4bp_equations, tspant1, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)

xvel = 0
yvel = 0

x = solT0.y[0,:]
y = solT0.y[1,:]
z = solT0.y[2,:]
# vx = solT0.y[3,:]
# vy = solT0.y[4,:]
t = solT0.t

for i in range(1,len(solT0.y[0,:])):

    distance = (x[i] - (1 - mu))**2 + y[i]**2
    # xyplanedistance = np.abs(z[i])
    xyplanecross = (z[i-1] * z[i]) < 0

    if distance < moondistSQ:
        # Only keeps the last place crossing
        if xyplanecross:

            # xend, yend, zend = x[i], y[i], z[i]
            tend = t[i]
            # print(i, xend, yend)

# Here
# print(i, xend, yend, tend)

tspant2 = (0,tend) # for 0 z position
solT1 = solve_ivp(bcr4bp_equations, tspant2, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)

x = solT1.y[0,:]
xend = x[-1]
y = solT1.y[1,:]
yend = y[-1]
z = solT1.y[2,:]
# Should be 0 or close enough
vx = solT1.y[3,:]
vxend = vx[-1]
vy = solT1.y[4,:]
vyend = vy[-1]
vz = solT1.y[5,:]
vzend = vz[-1] 
# print(vzend)

newstate1 = solT1.y[:,-1] + [0, 0, 0, 0, vyoffset, -vzend]
tspant3 = (tend,tend + 3)
deltav1 = np.sqrt(vyoffset**2 + vzend**2)

solT2 = solve_ivp(bcr4bp_equations, tspant3, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol, events = DRO_event)
x = solT2.y[0,:]
y = solT2.y[1,:]
z = solT2.y[2,:]
vx = solT2.y[3,:]
vy = solT2.y[4,:]
vz = solT2.y[5,:]
t2 = solT2.t[-1]


# Check if trajectory off the end intersects with DRO
r = []
for j in range(0,len(sol0_3BPDRO.y[0,:])):
    trajectorydistance = np.sqrt((x[-1] - sol0_3BPDRO.y[0,j])**2 + (y[-1] - sol0_3BPDRO.y[1,j])**2 + (z[-1] - sol0_3BPDRO.y[2,j])**2)
    r.append((j, trajectorydistance))

cpa = min(r, key=lambda e: e[1])
j, cpavalue = cpa
checkdistance = 5e-3

if cpavalue < checkdistance:
    endpoint = (x[-1], y[-1], z[-1])
    endtime = t2

    deltav2 = np.sqrt((-vx[-1]+sol0_3BPDRO.y[3,j])**2 + (-vy[-1]+sol0_3BPDRO.y[4,j])**2)

    deltav = deltav1 + deltav2
    # print('  deltav1: ', deltav1, 'DU/TU')
    # print('  deltav2: ', deltav2, 'DU/TU')
    DUtokm = 384.4e3 # kms in 1 DU
    TUtoS4 = 406074.761647 # s in 1 4BP TU
    deltavS = deltav * DUtokm / TUtoS4
    print('  deltavS: ', deltavS, 'km/s')
    print('  tend: ', t2, 'TU')
    tendS = t2 * TUtoS4
    tendday = tendS / (3600 * 24)
    print('  tend: ', tendday, 'days')


else:
    # pass
    while cpavalue > checkdistance:
        moonx = xend - (1-mu)
        moony = yend
        moonangle = np.arctan2(moony,moonx)
        # print('  moonangle:',moonangle)
        xvel += .05*np.cos(moonangle - thrustangle)         # Try different angles here
        yvel += .05*np.sin(moonangle - thrustangle)
        newstate1 = solT1.y[:,-1] + [0, 0, 0, xvel, vyoffset + yvel, -vzend]
        tspant3 = (tend,tend + 3)
        
        solT2 = solve_ivp(bcr4bp_equations, tspant3, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol, events = DRO_event)
        x = solT2.y[0,:]
        y = solT2.y[1,:]
        z = solT2.y[2,:]
        vx = solT2.y[3,:]
        vy = solT2.y[4,:]
        vz = solT2.y[5,:]


        # Check if trajectory off the end intersects with DRO
        r = []
        for i in range(0,len(x)):
            for j in range(0,len(sol0_3BPDRO.y[0,:])):
                trajectorydistance = np.sqrt((x[i] - sol0_3BPDRO.y[0,j])**2 + (y[i] - sol0_3BPDRO.y[1,j])**2 + (z[i] - sol0_3BPDRO.y[2,j])**2)
                r.append((i, j, trajectorydistance))

        cpa = min(r, key=lambda e: e[2])
        i, j, cpavalue = cpa
        t2 = solT2.t[i]


        deltav1 = np.sqrt(xvel**2 + (vyoffset + yvel)**2 + vzend**2)
        deltav2 = np.sqrt((-vx[i]+sol0_3BPDRO.y[3,j])**2 + (-vy[i]+sol0_3BPDRO.y[4,j])**2)
        deltav = deltav1 + deltav2
        DUtokm = 384.4e3 # kms in 1 DU
        TUtoS4 = 406074.761647 # s in 1 4BP TU
        deltavS = deltav * DUtokm / TUtoS4
        print('  deltavS: ', deltavS, 'km/s')
        print('  tend: ', t2, 'TU')
        tendS = t2 * TUtoS4
        tendday = tendS / (3600 * 24)
        print('  tend: ', tendday, 'days')




# 3D Plotting
# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Plot the celestial bodies
# # ax.scatter(-mu, 0, 0, color='blue', label='Earth', s=100)  # Primary body (Earth)
ax.scatter(1 - mu, 0, 0, color='gray', s = 30) #, label='Moon')  # Secondary body (Moon)

# # Plot the Lagrange points
ax.scatter([L1_x], [0], [0], color=[0.4660, 0.6740, 0.1880], s=15) #, label='L1')
ax.scatter([L2_x], [0], [0], color=[0.3010, 0.7450, 0.9330], s=15) #, label='L2')

# # ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=15, label='Langrage Points')

# # Plot the trajectories
ax.plot(sol0_3BPNRHO.y[0], sol0_3BPNRHO.y[1], sol0_3BPNRHO.y[2], color=[0, 0.4470, 0.7410]) #, label='9:2 NRHO')
ax.plot(sol0_3BPDRO.y[0], sol0_3BPDRO.y[1], sol0_3BPDRO.y[2], color=[0.4940, 0.1840, 0.5560]) #, label='70000km DRO')
# ax.plot(sol0_DROfind.y[0], sol0_DROfind.y[1], sol0_DROfind.y[2], color=[0.4940, 0.1840, 0.5560], label='Target DRO')

# ax.plot(solT0.y[0], solT0.y[1], solT0.y[2], color=[0.9290, 0.6940, 0.1250], label='T 1')
# ax.scatter([newstate1[0]], [newstate1[1]], [newstate1[2]], color=[0.8500, 0.3250, 0.0980], s=10)
ax.plot(solT1.y[0], solT1.y[1], solT1.y[2], color=[0.9290, 0.6940, 0.1250]) #, label='Coast Trajectory')

# ax.scatter([newstate3[0]], [newstate3[1]], [newstate3[2]], color=[0.8500, 0.3250, 0.0980], s=10)
ax.plot(solT2.y[0], solT2.y[1], solT2.y[2], color=[0.4660, 0.6740, 0.1880], label='T2')

# # ax.scatter([newstate4[0]], [newstate4[1]], [newstate4[2]], color=[0.8500, 0.3250, 0.0980], s=10)
# ax.plot(solT3.y[0], solT3.y[1], solT3.y[2], color=[0.4660, 0.6740, 0.1880]) #, label='DRO Intercept') # [0.9290, 0.6940, 0.1250]


# moonx = xend - (1-mu)
# moony = yend
# moonangle = np.arctan2(moony,moonx)
# xplot0 = .08*np.cos(moonangle)
# yplot0 = .08*np.sin(moonangle)
# xplot15 = .08*np.cos(moonangle - np.pi/12)
# yplot15 = .08*np.sin(moonangle - np.pi/12)
# xplot30 = .08*np.cos(moonangle - np.pi/6)
# yplot30 = .08*np.sin(moonangle - np.pi/6)
# xplot45 = .08*np.cos(moonangle - np.pi/4)
# yplot45 = .08*np.sin(moonangle - np.pi/4)
# xplot60 = .08*np.cos(moonangle - np.pi/3)
# yplot60 = .08*np.sin(moonangle - np.pi/3)
# xplot75 = .08*np.cos(moonangle - 5*np.pi/12)
# yplot75 = .08*np.sin(moonangle - 5*np.pi/12)
# xplot90 = .08*np.cos(moonangle - np.pi/2)
# yplot90 = .08*np.sin(moonangle - np.pi/2)

# ax.quiver(newstate1[0],newstate1[1],newstate1[2], xplot0, yplot0, 0, length = 1, color=[0.6350, 0.0780, 0.1840], label='0 Degrees')
# ax.quiver(newstate1[0],newstate1[1],newstate1[2], xplot15, yplot15, 0, length = 1, color=[0.3010, 0.7450, 0.9330], label='15 Degrees')
# ax.quiver(newstate1[0],newstate1[1],newstate1[2], xplot30, yplot30, 0, length = 1, color=[0.9290, 0.6940, 0.1250], label='30 Degrees')
# ax.quiver(newstate1[0],newstate1[1],newstate1[2], xplot45, yplot45, 0, length = 1, color=[0, 0.4470, 0.7410], label='45 Degrees')
# ax.quiver(newstate1[0],newstate1[1],newstate1[2], xplot60, yplot60, 0, length = 1, color=[0.8500, 0.3250, 0.0980], label='60 Degrees')
# ax.quiver(newstate1[0],newstate1[1],newstate1[2], xplot75, yplot75, 0, length = 1, color=[0.4660, 0.6740, 0.1880], label='75 Degrees')
# ax.quiver(newstate1[0],newstate1[1],newstate1[2], xplot90, yplot90, 0, length = 1, color=[0.4940, 0.1840, 0.5560], label='90 Degrees')

zticks = -.15, -.1, -.05, 0, .05

# Labels and plot settings
ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')
ax.set_zticks(zticks)
# ax.set_axis_off()  # Turn off the axes for better visual appeal

# ax.legend(loc='best')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()



