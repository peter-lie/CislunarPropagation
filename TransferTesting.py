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
DROx = sol0_3BPDRO.y[0,:]
DROy = sol0_3BPDRO.y[1,:]
DROz = sol0_3BPDRO.y[2,:]
DROvx = sol0_3BPDRO.y[3,:]
DROvy = sol0_3BPDRO.y[4,:]
DROvz = sol0_3BPDRO.y[5,:]


# sol1_4BPNRHO = solve_ivp(bcr4bp_equations, t_span2, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
# sol1_4BPDRO = solve_ivp(bcr4bp_equations, t_span2, state1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)


# Need new equations of motion to define constant thrust scenarios
thrust = 566.3e-3 # N
massSC = 39000 # kg, mass of gateway
Isp = 2517 # s
g0 = 9.80665 # m/s^2


# Constant Thrust Equations of Motion
def bcr4bp_constantthrust_equations_antivelocity(t, state, mu, inc, Omega, theta0, thrust):
    # Unpack the state vector
    x, y, z, vx, vy, vz, massSC = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

    # Requres thrust, Isp, g0, and mass for true differential equation
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)

    xThrust = 0 - (thrust/massSC) * (vx/velocity)         # Use + (T/m)*(dx/v) for with vvec
    yThrust = 0 - (thrust/massSC) * (vy/velocity)         # Use - (T/m)*(dx/v) against
    zThrust = 0 - (thrust/massSC) * (vz/velocity)    

    # Thrust = (N/kg) * (DU/TU * TU/DU) = m/s^2

    # All in m/s^2, require DU/TU^2
    DUtom = 384.4e6 # m in 1 DU
    TUtoS4 = 406074.761647 # s in 1 4BP TU
    xThrust = xThrust / DUtom * TUtoS4**2
    yThrust = yThrust / DUtom * TUtoS4**2
    zThrust = zThrust / DUtom * TUtoS4**2

    # Accelerations from the Sun's gravity (transformed)
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t, inc, Omega, theta0)

    Isp = 2517 # s
    g0 = 9.80665 # m/s^2
    dmass = - thrust / (Isp * g0) * TUtoS4 # all in SI units 

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + a_Sx + xThrust
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3 + a_Sy + yThrust
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3 + a_Sz + zThrust

    return [vx, vy, vz, ax, ay, az, dmass]

# Unused, not sure about validity of equations (control law)
def bcr4bp_constantthrust_equations_control(t, state, mu, inc, Omega, theta0, thrust):
    # Unpack the state vector
    x, y, z, vx, vy, vz, massSC = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

    # Requres thrust, Isp, g0, and mass for true differential equation
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)

    # Control law ideas: have the DRO points within the differential equation (should have as data, not calculate every time)
    # Include a term for matching the velocity of the closest point of the DRO
    # Include a term point from the current position to the closest position of the DRO

    # Find closest point, norm, combine

    # Find closest point to DRO
    r = []
    for j in range(0,len(DROx)):
        # trajectorydistance = np.sqrt((x[i] - sol0_3BPDRO.y[0,j])**2 + (y[i] - sol0_3BPDRO.y[1,j])**2 + (z[i] - sol0_3BPDRO.y[2,j])**2)
        trajectorydistance = np.sqrt((x - DROx[j])**2 + (y - DROy[j])**2 + (z - DROz[j])**2) # not using z distance
        r.append((j, trajectorydistance))

    cpa = min(r, key=lambda e: e[1])
    j, cpavalue = cpa

    # v1direction = [DRO(vx), DRO(vy), DRO(vz)]
    velocityerror = [DROvx[j] - vx, DROvy[j] - vy, DROvz[j] - vz]  # direction of DRO velocity
    # velocityerror = velocityerror/np.linalg.norm(velocityerror)
    # v2direction = [x - DRO(x), y - DRO(y), z - DRO(z)]
    positionerror = [(1-mu) - x, 0 - y, 0 - z]
    # positionerror = positionerror/np.linalg.norm(positionerror)
    # Combine directions together, need to change these weights depending
    direction = 0 * np.array(velocityerror) + 1 * np.array(positionerror)
    direction = direction/np.linalg.norm(direction)

    xThrust = 0 + (thrust/massSC) * direction[0]   
    yThrust = 0 + (thrust/massSC) * direction[1]         
    zThrust = 0 + (thrust/massSC) * direction[2]  

    # Thrust = (N/kg) * (DU/TU * TU/DU) = m/s^2

    # All in m/s^2, require DU/TU^2
    DUtom = 384.4e6 # m in 1 DU
    TUtoS4 = 406074.761647 # s in 1 4BP TU
    xThrust = xThrust / DUtom * TUtoS4**2
    yThrust = yThrust / DUtom * TUtoS4**2
    zThrust = zThrust / DUtom * TUtoS4**2

    # Accelerations from the Sun's gravity (transformed)
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t, inc, Omega, theta0)

    Isp = 2517 # s
    g0 = 9.80665 # m/s^2
    dmass = - thrust / (Isp * g0) * TUtoS4 # all in SI units 

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + a_Sx + xThrust
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3 + a_Sy + yThrust
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3 + a_Sz + zThrust

    return [vx, vy, vz, ax, ay, az, dmass]

# Continuous thrust in 3BP
def cr3bp_constantthrust_equations(t, state, mu, thrust):
    # Unpack the state vector
    x, y, z, vx, vy, vz, massSC = state
    
    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

    # Requres thrust, Isp, g0, and mass for true differential equation
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)

    # Find closest point to DRO
    r = []
    for j in range(0,len(DROx)):
        # trajectorydistance = np.sqrt((x[i] - sol0_3BPDRO.y[0,j])**2 + (y[i] - sol0_3BPDRO.y[1,j])**2 + (z[i] - sol0_3BPDRO.y[2,j])**2)
        trajectorydistance = np.sqrt((x - DROx[j])**2 + (y - DROy[j])**2 + (z - DROz[j])**2) # not using z distance
        r.append((j, trajectorydistance))

    cpa = min(r, key=lambda e: e[1])
    j, cpavalue = cpa

    # v1direction = [DRO(vx), DRO(vy), DRO(vz)]
    velocityerror = [DROvx[j] - vx, DROvy[j] - vy, DROvz[j] - vz]  # direction of DRO velocity
    # velocityerror = velocityerror/np.linalg.norm(velocityerror)
    # v2direction = [x - DRO(x), y - DRO(y), z - DRO(z)]
    positionerror = [DROx[j] - x, DROy[j] - y, DROz[j] - z]
    # positionerror = positionerror/np.linalg.norm(positionerror)
    # Combine directions together, need to change these weights depending
    direction = 1 * np.array(velocityerror) + 0 * np.array(positionerror)
    direction = direction/np.linalg.norm(direction)

    xThrust = 0 + (thrust/massSC) * direction[0]   
    yThrust = 0 + (thrust/massSC) * direction[1]         
    zThrust = 0 + (thrust/massSC) * direction[2]

    # xThrust = 0 + (thrust/massSC) * (vx/velocity)         # Use + (T/m)*(dx/v) for with vvec
    # yThrust = 0 + (thrust/massSC) * (vy/velocity)         # Use - (T/m)*(dx/v) against
    # zThrust = 0 - (thrust/massSC) * (vz/velocity)    

    # All in m/s^2, require DU/TU^2
    DUtom = 384.4e6 # m in 1 DU
    TUtoS3 = 375401.94737 # s in 1 4BP TU
    xThrust = xThrust / DUtom * TUtoS3**2
    yThrust = yThrust / DUtom * TUtoS3**2
    zThrust = zThrust / DUtom * TUtoS3**2

    Isp = 2517 # s
    g0 = 9.80665 # m/s^2
    dmass = - thrust / (Isp * g0) * TUtoS3 # all in SI units 

    # Equations of motion
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3 + xThrust
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3 + yThrust
    az = -(1 - mu)*z/r1**3 - mu*z/r2**3 + zThrust
    
    return [vx, vy, vz, ax, ay, az, dmass]


# Hypothetical transfer maneuvers
# Starting with 3BP NRHO characteristics, looking for 3BP DRO characteristics

# Loop to check for the last time orbit crosses the xy plane inside of the DRO

theta0 = 0
thetastep = np.pi * 3 # 32 points takes 10 minutes to run, 128 points takes 40 minutes
# thetastep = np.pi/256
thetamax = 2 * np.pi
deltavmin = 1
thetamin = 0

moondistSQ = (1*(moondist/384.4e3))**2
deltavstorage = {}
# distancecheck = {}
# distancecheck[0] = 0.1

# state0CT = state0

# while theta0 < thetamax:
print('theta0: ', theta0)
# Let run for 10 TU first, scale back on x and y
massSC = 39000 # set mass back to starting value
tspant0 = (0,0) # letting the spacecraft move around before burning - often results in added z velocity
# solPreburn = solve_ivp(bcr4bp_equations, tspant0, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
# state1CT = solPreburn.y[:,-1]
# state1CT = [state1CT[0], state1CT[1], state1CT[2], state1CT[3], state1CT[4], state1CT[5], massSC]
state1CT = [state0[0], state0[1], state0[2], state0[3], state0[4], state0[5], massSC]

# print('  state1CT: ', state1CT)

tspant1 = (tspant0[1],10) # for DRO x-y intersection
# solT0 = solve_ivp(bcr4bp_constantthrust_equations_antivelocity, tspant1, state1CT, args=(mu,inc,Omega0,theta0,thrust,), rtol=tol, atol=tol)
solT0 = solve_ivp(cr3bp_constantthrust_equations, tspant1, state1CT, args=(mu,thrust,), rtol=tol, atol=tol)
x = solT0.y[0,:]
y = solT0.y[1,:]
z = solT0.y[2,:]
vx = solT0.y[3,:]
vy = solT0.y[4,:]
m = solT0.y[6,:]

t = solT0.t

# # Check if trajectory off the end intersects with DRO
# r = []
# for i in range(0,len(x)):
#     for j in range(0,len(sol0_3BPDRO.y[0,:])):
#         # trajectorydistance = np.sqrt((x[i] - sol0_3BPDRO.y[0,j])**2 + (y[i] - sol0_3BPDRO.y[1,j])**2 + (z[i] - sol0_3BPDRO.y[2,j])**2)
#         trajectorydistance = np.sqrt((x[i] - DROx[j])**2 + (y[i] - DROy[j])**2) # not using z distance
#         r.append((i, j, trajectorydistance))

# cpa = min(r, key=lambda e: e[2])
# i, j, cpavalue = cpa
# checkdistance = 1e-2

# if cpavalue < checkdistance:
#     endpoint = (x[i], y[i], z[i])
#     endtime = t[i]
#     print(  'endtime: ',endtime)
#     tspant2 = (tspant0[1],endtime)
#     solT1 = solve_ivp(bcr4bp_constantthrust_equations_antivelocity, tspant2, state1CT, args=(mu,inc,Omega0,theta0,thrust,), rtol=tol, atol=tol)
#     solT4 = solve_ivp(bcr4bp_equations, tspant2, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
#     deltav1 = np.sqrt((-vx[i]+DROvx[j])**2 + (-vy[i]+DROvy[j])**2)
#     state2CT = solT1.y[:,-1] + [0, 0, 0, 0, 0, 0, 0]
#     # state2 = state2CT[0:6]
#     tspant3 = (tspant2[1], tspant2[1]+10) # should always cross xy plane

#     # solT2 = solve_ivp(bcr4bp_equations, tspant3, state2, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
#     solT2 = solve_ivp(bcr4bp_constantthrust_equations_control, tspant3, state2CT, args=(mu,inc,Omega0,theta0,thrust,), rtol=tol, atol=tol)
#     solT3 = solve_ivp(bcr4bp_constantthrust_equations_antivelocity, tspant3, state2CT, args=(mu,inc,Omega0,theta0,thrust,), rtol=tol, atol=tol)

    # z = solT2.y[2,:]
    # t = solT2.t

    # for i in range(1,len(solT2.y[0,:])):
    #     xyplanecross = (z[i-1] * z[i]) < 0
    #     if xyplanecross:
    #         # xend, yend, zend = x[i], y[i], z[i]
    #         tend = t[i]
    #         # print(i, xend, yend)

    # tspant4 = (tspant2[1], tend)
    # solT3 = solve_ivp(bcr4bp_equations, tspant4, state2, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
    # state3 = solT3.y[:,-1] + [0, 0, 0, 0, 0, -solT3.y[5,-1]]
    # deltav2 = np.sqrt(solT3.y[5,-1]**2)
    # # tspant5 = (tspant4[1], tspant4[1]+3)
    # # solT4 = solve_ivp(bcr4bp_equations, tspant5, state3, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)


exitvelocity = 24.69 # km/s
massratio = m[0]/m[-1]
deltavtrue = exitvelocity * np.log(massratio)

# deltav = deltav1 # + deltav2
# print('  deltav1: ', deltav1, 'DU/TU')
# print('  deltav2: ', deltav2, 'DU/TU')
DUtokm = 384.4e3 # kms in 1 DU
TUtoS4 = 406074.761647 # s in 1 4BP TU
# deltavS = deltav * DUtokm / TUtoS4

deltavtotal = deltavtrue #+ deltavS # fix this into the right units
print('  initial mass:', m[0], ' km/s')
print('  final mass:', m[-1], ' km/s')
print('  total deltav:', deltavtotal, ' km/s')

deltavstorage[theta0] = deltavtotal
if deltavtotal < deltavmin:
    deltavmin = deltavtotal
    thetamin = theta0


# theta0 += thetastep


# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot Moon, Lagrange Points
ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=30)  # Secondary body (Moon)
ax.scatter([L1_x], [0], [0], color=[0.4660, 0.6740, 0.1880], s=15, label='L1')
ax.scatter([L2_x], [0], [0], color=[0.3010, 0.7450, 0.9330], s=15, label='L2')

# Plot the trajectories
ax.plot(sol0_3BPNRHO.y[0], sol0_3BPNRHO.y[1], sol0_3BPNRHO.y[2], color=[0, 0.4470, 0.7410], label='9:2 NRHO')
ax.plot(sol0_3BPDRO.y[0], sol0_3BPDRO.y[1], sol0_3BPDRO.y[2], color=[0.4940, 0.1840, 0.5560], label='70000km DRO')

ax.plot(solT0.y[0], solT0.y[1], solT0.y[2], color=[0.9290, 0.6940, 0.1250], label='Thrust')
# ax.plot(solT1.y[0], solT1.y[1], solT1.y[2], color=[0.9290, 0.6940, 0.1250], label='Thrust')
#ax.plot(solT2.y[0], solT2.y[1], solT2.y[2], color=[0.4660, 0.6740, 0.1880], label='Control')
# ax.scatter([newstate1[0]], [newstate1[1]], [newstate1[2]], color=[0.8500, 0.3250, 0.0980], s=10, label='Maneuver')
# ax.plot(solT3.y[0], solT3.y[1], solT3.y[2], color=[0.8500, 0.3250, 0.0980], label='AV')
# ax.plot(solT4.y[0], solT4.y[1], solT4.y[2], color=[0.4660, 0.6740, 0.1880], label='Coast')

# Labels and plot settings
ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')
ax.set_title(f'Solar Theta0: {theta0}')

ax.legend(loc='best')
plt.show()



print('     deltavmin:', deltavmin)
print('     @theta0:', thetamin)

# plt.figure(figsize=(10, 6))
# plt.plot(deltavstorage.keys(), deltavstorage.values())

# plt.xlabel('Solar Theta0 [rads]')
# plt.ylabel('deltaV [km/s]')
# plt.title('deltaV vs Theta0')

# plt.show()

