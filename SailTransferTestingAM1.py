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
from functools import partial
# from mpl_toolkits.mplot3d import Axes3D

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

moondistSQ = ((1 - mu - state1[0]))**2
# print(moondistSQ)

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

# Unused, not sure about validity of equations (control law)
def bcr4bp_solarsail_equations_againstZ(t, state, mu, inc, Omega, theta0):
    # Unpack the state vector
    x, y, z, vx, vy, vz = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

        # Solar Position
    r_Sx0, r_Sy0, r_Sz0 = sun_position(t, inc, Omega0, theta0)
    sunDistance = np.sqrt(r_Sx0**2 + r_Sy0**2 + r_Sz0**2)

    cr = 1.2
    Psrp = 4.57e-6 # Pa
    Amratio = 1 # m^2/kg
    # Amratio = 4.8623877 # m^2/kg
    SF = 1 # assume always in sun (NRHO designed for this)

    aSRPx = - Psrp * cr * (Amratio) * SF * ((r_Sx0-x) / sunDistance)
    aSRPy = - Psrp * cr * (Amratio) * SF * ((r_Sy0-y) / sunDistance)
    aSRPz = - Psrp * cr * (Amratio) * SF * ((r_Sz0-z) / sunDistance)

    # All in m/s^2, require DU/TU^2
    DUtom = 384.4e6 # m in 1 DU
    TUtoS4 = 406074.761647 # s in 1 4BP TU
    aSRPx = aSRPx / DUtom * TUtoS4**2
    aSRPy = aSRPy / DUtom * TUtoS4**2
    aSRPz = aSRPz / DUtom * TUtoS4**2

    zforvelocity = (aSRPz * vz) > 0

    if zforvelocity:
        aSRPx = 0
        aSRPy = 0
        aSRPz = 0

    # Accelerations from the Sun's gravity (transformed)
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t, inc, Omega, theta0)

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + a_Sx + aSRPx
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3 + a_Sy + aSRPy
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3 + a_Sz + aSRPz

    return [vx, vy, vz, ax, ay, az]


def bcr4bp_solarsail_equations_withXY(t, state, mu, inc, Omega, theta0):
    # Unpack the state vector
    x, y, z, vx, vy, vz = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

        # Solar Position
    r_Sx0, r_Sy0, r_Sz0 = sun_position(t, inc, Omega0, theta0)
    sunDistance = np.sqrt(r_Sx0**2 + r_Sy0**2 + r_Sz0**2)

    cr = 1.2
    Psrp = 4.57e-6 # Pa
    Amratio = 1 # m^2/kg
    # Amratio = 4.8623877 # m^2/kg
    SF = 1 # assume always in sun (NRHO designed for this, DRO close enough)

    aSRPx = - Psrp * cr * (Amratio) * SF * ((r_Sx0-x) / sunDistance)
    aSRPy = - Psrp * cr * (Amratio) * SF * ((r_Sy0-y) / sunDistance)
    aSRPz = - Psrp * cr * (Amratio) * SF * ((r_Sz0-z) / sunDistance)

    # All in m/s^2, require DU/TU^2
    DUtom = 384.4e6 # m in 1 DU
    TUtoS4 = 406074.761647 # s in 1 4BP TU
    aSRPx = aSRPx / DUtom * TUtoS4**2
    aSRPy = aSRPy / DUtom * TUtoS4**2
    aSRPz = aSRPz / DUtom * TUtoS4**2

    if r2 < .0105:
        aSRPx = 0
        aSRPy = 0
        aSRPz = 0

    xagainstvelocity = (aSRPx * vx) < 0

    if xagainstvelocity:
        aSRPx = 0
        aSRPy = 0
        aSRPz = 0

    yagainstvelocity = (aSRPy * vy) < 0

    if yagainstvelocity:
        aSRPx = 0
        aSRPy = 0
        aSRPz = 0

    # Accelerations from the Sun's gravity (transformed)
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t, inc, Omega, theta0)

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + a_Sx + aSRPx
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3 + a_Sy + aSRPy
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3 + a_Sz + aSRPz

    return [vx, vy, vz, ax, ay, az]


def bcr4bp_solarsail_equations_againstXY(t, state, mu, inc, Omega, theta0):
    # Unpack the state vector
    x, y, z, vx, vy, vz = state

    # Distances to primary and secondary
    r1, r2 = r1_r2(x, y, z, mu)

        # Solar Position
    r_Sx0, r_Sy0, r_Sz0 = sun_position(t, inc, Omega0, theta0)
    sunDistance = np.sqrt(r_Sx0**2 + r_Sy0**2 + r_Sz0**2)

    cr = 1.2
    Psrp = 4.57e-6 # Pa
    Amratio = 1 # m^2/kg
    # Amratio = 4.8623877 # m^2/kg
    SF = 1 # assume always in sun (NRHO designed for this, DRO close enough)

    aSRPx = - Psrp * cr * (Amratio) * SF * ((r_Sx0-x) / sunDistance)
    aSRPy = - Psrp * cr * (Amratio) * SF * ((r_Sy0-y) / sunDistance)
    aSRPz = - Psrp * cr * (Amratio) * SF * ((r_Sz0-z) / sunDistance)

    # All in m/s^2, require DU/TU^2
    DUtom = 384.4e6 # m in 1 DU
    TUtoS4 = 406074.761647 # s in 1 4BP TU
    aSRPx = aSRPx / DUtom * TUtoS4**2
    aSRPy = aSRPy / DUtom * TUtoS4**2
    aSRPz = aSRPz / DUtom * TUtoS4**2

    if r2 < .0105:
        aSRPx = 0
        aSRPy = 0
        aSRPz = 0

    xagainstvelocity = (aSRPx * vx) > 0

    if xagainstvelocity:
        aSRPx = 0
        aSRPy = 0
        aSRPz = 0

    yagainstvelocity = (aSRPy * vy) > 0

    if yagainstvelocity:
        aSRPx = 0
        aSRPy = 0
        aSRPz = 0

    # Accelerations from the Sun's gravity (transformed)
    a_Sx, a_Sy, a_Sz = sun_acceleration(x, y, z, t, inc, Omega, theta0)

    # Full equations of motion with Coriolis and Sun's effect
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + a_Sx + aSRPx
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3 + a_Sy + aSRPy
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3 + a_Sz + aSRPz

    return [vx, vy, vz, ax, ay, az]


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
        if distance < .0002:
            # See if greater than that point
    
            distunder = (circleplotx[i]-x) + (circleploty[i]-y)

    # Cross from positive to negative
    # output = moondistSQ - distance
    output = distunder
    # print(output)

    return output


# Ensure the solver stops at the event
# event_DROintercept.terminal = True  
# event_DROintercept.direction = 0  # Detect both approaching and receding


# Hypothetical transfer maneuvers
# Starting with 3BP NRHO characteristics, looking for 3BP DRO characteristics

# Loop to check for the last time orbit crosses the xy plane inside of the DRO

# Am = .001, theta0 = 3.8288160465625496, deltav = 0.40493730289588753
# Am = .005, theta0 = 1.6812429435226592, deltav = 0.4313581060787977
# Am = .01, theta0 = 4.822835597112472, deltav = 0.4258592250225026
# Am = .05, theta0 = 3.742913122440954, deltav = 0.36926246709170213
# Am = .07, theta0 = 2.073942025221382, deltav = 0.41446372408631577
# Am = .1, theta0 = 1.6935147898257443, deltav = 0.3759867211358866
# Am = .5, theta0 = 0.1595340019401067, deltav = 0.4019372064876697
# Am = .75, theta0 = 2.049398332615212, deltav = 0.4099516793670503
# Am = 1, theta0 = 1.914408023281276, deltav = 0.3403664504612836
# Am = 2, theta0 = 5.154175447295781, deltav = 0.2920858816032296       (error)
# Am = 5, theta0 = 0.9817477042468091, deltav = 0.4262960566246122
# Am = 10, theta0 = 0.06135923151542565, deltav = 0.37346585304863633
# Am = 15, theta0 = 5.105088062083439, deltav = 0.39902776634298043
# Am = 20, theta0 = 5.301437602932808, deltav = 0.42094563555352793     (error)

# theta0 = 4.319689898685965, deltaV = 0.3833353267993585 bit of an error probably
# theta0 = 1.914408023281276, deltav = 0.3403664504612836 try this one for return

# theta0 = 3.3624858870453163, deltav = 0.36023662397180306 try this one for return
# theta0 = 2.1598449493429777, deltav = 0.3411132348001378 bit of an error probably



# Change to desired angle
theta0 = 1.914408023281276
theta0 = 3.3624858870453163 # in Example 2 file
xoffset = 0


tspant1 = (0,21) # for DRO x-y intersection
# solT0 = solve_ivp(bcr4bp_constantthrust_equations_antivelocity, tspant1, state1CT, args=(mu,inc,Omega0,theta0,thrust,), rtol=tol, atol=tol)
solT0 = solve_ivp(bcr4bp_solarsail_equations_againstZ, tspant1, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
x = solT0.y[0,:]
y = solT0.y[1,:]
z = solT0.y[2,:]
vx = solT0.y[3,:]
vy = solT0.y[4,:]

t = solT0.t

for i in range(1,len(solT0.y[0,:])):

    distance = (x[i] - (1 - mu))**2 + y[i]**2
    xyplanecross = (z[i-1] * z[i]) < 0

    # if distance < moondistSQ:
    # Only keeps the last place crossing
    if xyplanecross:
        tend = t[i]

# print(' tend:',tend)

tspant2 = (0,tend) # for 0 z position
solT1 = solve_ivp(bcr4bp_solarsail_equations_againstZ, tspant2, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
xend = solT1.y[0,-1] 
yend = solT1.y[1,-1]

vzend = solT1.y[5,-1] 
# print(vzend)

newstate1 = solT1.y[:,-1] + [0, 0, 0, xoffset, 0, -(vzend)]
tspant3 = (tend,tend + 3)  # Chance here to let trajectory try longer or shorter
# tspant3 = (tend,tend + 0)  # Chance here to let trajectory try longer or shorter
deltav1 = np.sqrt(vzend**2 + xoffset**2)

# Now on XY plane, need to get out to DRO
# Solve with the event function

solT2 = solve_ivp(bcr4bp_solarsail_equations_withXY, tspant3, newstate1, args=(mu, inc, Omega0, theta0), rtol=tol, atol=tol, events = DRO_event)
# solT2 = solve_ivp(bcr4bp_equations, tspant3, newstate1, args=(mu, inc, Omega0, theta0), rtol=tol, atol=tol, events = DRO_event)

# newstate2 = solT2.y[:,-1]
# tspant4 = (tspant3[1],tspant3[1] + 9)  # Chance here to let trajectory try longer or shorter

# # Now on XY plane, need to get out to DRO
# # Solve with the event function

# # solT3 = solve_ivp(bcr4bp_solarsail_equations_withXY, tspant4, newstate2, args=(mu, inc, Omega0, theta0), rtol=tol, atol=tol, events = DRO_event)
# solT3 = solve_ivp(bcr4bp_equations, tspant4, newstate2, args=(mu, inc, Omega0, theta0), rtol=tol, atol=tol, events = DRO_event)


x = solT2.y[0,:]
xend = x[-1]
y = solT2.y[1,:]
yend = y[-1]
vx = solT2.y[3,:]
vxend = vx[-1]
vy = solT2.y[4,:]
vyend = vy[-1]
tend3 = solT2.t[-1]

# newstate3 = solT3.y[:,-1]
# tspant5 = (tend3,tend3 + 8)

# # solT4 = solve_ivp(bcr4bp_solarsail_equations_againstXY, tspant5, newstate3, args=(mu, inc, Omega0, theta0), rtol=tol, atol=tol, events = DRO_event)
# # solT4 = solve_ivp(bcr4bp_equations, tspant5, newstate3, args=(mu, inc, Omega0, theta0), rtol=tol, atol=tol, events = DRO_event)
# solT4 = solve_ivp(bcr4bp_solarsail_equations_withXY, tspant5, newstate3, args=(mu, inc, Omega0, theta0), rtol=tol, atol=tol, events = DRO_event)

# x = solT4.y[0,:]
# xend = x[-1]
# y = solT4.y[1,:]
# yend = y[-1]
# vx = solT4.y[3,:]
# vxend = vx[-1]
# vy = solT4.y[4,:]
# vyend = vy[-1]
# tend4 = solT4.t[-1]

# Closest DRO point
r = []
for j in range(0,len(sol0_3BPDRO.y[0,:])):
    trajectorydistance = np.sqrt((xend - DROx[j])**2 + (yend - DROy[j])**2) # not using z distance
    # Implement velocity check as well
    # relativevelocity = np.sqrt((vxend - DROvx[j])**2 + (vyend - DROvy[j])**2)
    r.append((j, trajectorydistance))


cpa = min(r, key=lambda e: e[1])
j, cpavalue = cpa

deltav2 = np.sqrt( (DROvx[j] - vxend)**2 + (DROvy[j] - vyend)**2 )
deltav = deltav1 + deltav2
# print('  deltav1: ', deltav1, 'DU/TU')
# print('  deltav2: ', deltav2, 'DU/TU')
DUtokm = 384.4e3 # kms in 1 DU
TUtoS4 = 406074.761647 # s in 1 4BP TU
deltavS = deltav * DUtokm / TUtoS4
deltav1S = deltav1 * DUtokm / TUtoS4

print('  deltav1: ', deltav1S, 'km/s')
print('  deltavS: ', deltavS, 'km/s')

print('  tend:    ', tend3, 'TU')
tendS = tend3 * TUtoS4
tendday = tendS / (3600 * 24)
print('  tend:    ', tendday, 'days')


space = np.linspace(0,100)
circleplotx = np.sqrt(.035) * np.sin(2*np.pi*space/100) + (1-.68*mu)
circleploty = np.sqrt(.061) * np.cos(2*np.pi*space/100)

# MATLAB Default colors
# 1: [0, 0.4470, 0.7410]        Blue
# 2: [0.8500, 0.3250, 0.0980]   Red
# 3: [0.9290, 0.6940, 0.1250]   Yellow
# 4: [0.4940, 0.1840, 0.5560]   Purple
# 5: [0.4660, 0.6740, 0.1880]
# 6: [0.3010, 0.7450, 0.9330]
# 7: [0.6350, 0.0780, 0.1840]


r_Sx0, r_Sy0, r_Sz0 = sun_position(0, inc, Omega0, theta0)
r_Sx1, r_Sy1, r_Sz1 = sun_position(tend3, inc, Omega0, theta0)

# r_Sx1, r_Sy1, r_Sz1 = sun_position(tspant4[1], inc, Omega0, theta0)

sundist = np.sqrt(r_Sx0**2 + r_Sy0**2 + r_Sz0**2)


# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot Moon, Lagrange Points
ax.scatter(-mu, 0, 0, color='blue', label='Earth', s=80)  # Primary body (Earth)
ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=25)  # Secondary body (Moon)
ax.scatter([L1_x], [0], [0], color=[0.8500, 0.3250, 0.0980], s=10, label='L Points')
ax.scatter([L2_x], [0], [0], color=[0.8500, 0.3250, 0.0980], s=10)

# Plot the trajectories
# ax.plot(sol0_3BPNRHO.y[0], sol0_3BPNRHO.y[1], sol0_3BPNRHO.y[2], color=[0, 0.4470, 0.7410], label='9:2 NRHO')
ax.plot(sol0_3BPDRO.y[0], sol0_3BPDRO.y[1], sol0_3BPDRO.y[2], color=[0.4940, 0.1840, 0.5560], label='70000km DRO')

# ax.plot(solT0.y[0], solT0.y[1], solT0.y[2], color=[0.9290, 0.6940, 0.1250], label='Solar Sail')
# ax.plot(solT01.y[0], solT01.y[1], solT01.y[2], color=[0.4660, 0.6740, 0.1880], label='Coast')
ax.plot(solT1.y[0], solT1.y[1], solT1.y[2], color=[0.9290, 0.6940, 0.1250])
ax.plot(solT2.y[0], solT2.y[1], solT2.y[2], color=[0.4660, 0.6740, 0.1880])
# ax.scatter([newstate1[0]], [newstate1[1]], [newstate1[2]], color=[0.8500, 0.3250, 0.0980], s=10, label='Maneuver')
# ax.plot(solT3.y[0], solT3.y[1], solT3.y[2], color=[0.8500, 0.3250, 0.0980], label='DRO Connect')
# ax.plot(solT4.y[0], solT4.y[1], solT4.y[2], color=[0.4660, 0.6740, 0.1880], label='Coast')

# ax.plot(circleplotx,circleploty)

# Solar Positions
ax.quiver(-mu,0,0, r_Sx0/sundist, r_Sy0/sundist, r_Sz0/sundist, length = .5, color=[0,0,0], alpha = 1, label='Initial Sun Vector')
ax.quiver(-mu,0,0, r_Sx1/sundist, r_Sy1/sundist, r_Sz1/sundist, length = .5, color=[0,0,0], alpha = .5, label='Final SUn Vector')


zticks = -.2, 0, .2
# Labels and plot settings
ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')
ax.set_zticks(zticks)
ax.set_title(f'Solar Theta0: {theta0}')
ax.set_aspect('equal', adjustable='box')

# ax.legend()
# bbox_to_anchor=(100, 100)

plt.show()





# # Checking left hand turn by moon
# x = solT1.y[0]
# y = solT1.y[1]
# z = solT1.y[2]
# vx = solT1.y[3]
# vy = solT1.y[4]
# vz = solT1.y[5]
# t = solT1.t

# r1 = np.zeros(len(x))
# r2 = np.zeros(len(x))
# a_Sx = np.zeros(len(x))
# a_Sy = np.zeros(len(x))
# a_Sz = np.zeros(len(x))
# ax = np.zeros(len(x))
# ay = np.zeros(len(x))
# az = np.zeros(len(x))

# for i in range(0,len(x)):
#     # Distances to primary and secondary
#     r1[i], r2[i] = r1_r2(x[i], y[i], z[i], mu)
    
#     # Accelerations from the Sun's gravity (transformed)
#     a_Sx[i], a_Sy[i], a_Sz[i] = sun_acceleration(x[i], y[i], z[i], t[i], inc, Omega0, theta0)

#     # Full equations of motion with Coriolis and Sun's effect
#     ax[i] = 2 * vy[i] + x[i] - (1 - mu) * (x[i] + mu) / r1[i]**3 - mu * (x[i] - (1 - mu)) / r2[i]**3 + a_Sx[i]
#     ay[i] = -2 * vx[i] + y[i] - (1 - mu) * y[i] / r1[i]**3 - mu * y[i] / r2[i]**3 + a_Sy[i]
#     az[i] = -(1 - mu) * z[i] / r1[i]**3 - mu * z[i] / r2[i]**3 + a_Sz[i]


# x2 = solT2.y[0]
# y2 = solT2.y[1]
# z2 = solT2.y[2]
# vx2 = solT2.y[3]
# vy2 = solT2.y[4]
# vz2 = solT2.y[5]
# t2 = solT2.t

# r1_2 = np.zeros(len(x2))
# r2_2 = np.zeros(len(x2))
# a_Sx2 = np.zeros(len(x2))
# a_Sy2 = np.zeros(len(x2))
# a_Sz2 = np.zeros(len(x2))
# ax2 = np.zeros(len(x2))
# ay2 = np.zeros(len(x2))
# az2 = np.zeros(len(x2))

# for i in range(0,len(x2)):
#     # Distances to primary and secondary
#     r1_2[i], r2_2[i] = r1_r2(x2[i], y2[i], z2[i], mu)
    
#     # Accelerations from the Sun's gravity (transformed)
#     a_Sx2[i], a_Sy2[i], a_Sz2[i] = sun_acceleration(x2[i], y2[i], z2[i], t2[i], inc, Omega0, theta0)

#     # Full equations of motion with Coriolis and Sun's effect
#     ax2[i] = 2 * vy2[i] + x2[i] - (1 - mu) * (x2[i] + mu) / r1_2[i]**3 - mu * (x2[i] - (1 - mu)) / r2_2[i]**3 + a_Sx2[i]
#     ay2[i] = -2 * vx2[i] + y2[i] - (1 - mu) * y2[i] / r1_2[i]**3 - mu * y2[i] / r2_2[i]**3 + a_Sy2[i]
#     az2[i] = -(1 - mu) * z2[i] / r1_2[i]**3 - mu * z2[i] / r2_2[i]**3 + a_Sz2[i]




# x3 = solT3.y[0]
# y3 = solT3.y[1]
# z3 = solT3.y[2]
# vx3 = solT3.y[3]
# vy3 = solT3.y[4]
# vz3 = solT3.y[5]
# t3 = solT3.t

# r1_3 = np.zeros(len(x3))
# r2_3 = np.zeros(len(x3))
# a_Sx3 = np.zeros(len(x3))
# a_Sy3 = np.zeros(len(x3))
# a_Sz3 = np.zeros(len(x3))
# ax3 = np.zeros(len(x3))
# ay3 = np.zeros(len(x3))
# az3 = np.zeros(len(x3))
# ax4 = np.zeros(len(x3))
# ay4 = np.zeros(len(x3))
# az4 = np.zeros(len(x3))

# for i in range(0,len(x3)):
#     # Distances to primary and secondary
#     r1_3[i], r2_3[i] = r1_r2(x3[i], y3[i], z3[i], mu)
    
#     # Accelerations from the Sun's gravity (transformed)
#     a_Sx3[i], a_Sy3[i], a_Sz3[i] = sun_acceleration(x3[i], y3[i], z3[i], t3[i], inc, Omega0, theta0)

#     # Full equations of motion with Coriolis and Sun's effect
#     ax3[i] = 2 * vy3[i] + x3[i] - (1 - mu) * (x3[i] + mu) / r1_3[i]**3 - mu * (x3[i] - (1 - mu)) / r2_3[i]**3 + a_Sx3[i]
#     ay3[i] = -2 * vx3[i] + y3[i] - (1 - mu) * y3[i] / r1_3[i]**3 - mu * y3[i] / r2_3[i]**3 + a_Sy3[i]
#     az3[i] = -(1 - mu) * z3[i] / r1_3[i]**3 - mu * z3[i] / r2_3[i]**3 + a_Sz3[i]



# # 1: [0, 0.4470, 0.7410]        Blue
# # 2: [0.8500, 0.3250, 0.0980]   Red
# # 3: [0.9290, 0.6940, 0.1250]   Yellow
# # 4: [0.4940, 0.1840, 0.5560]   Purple


# # 2D Plotting to check Velocity
# plt.figure(figsize=(12, 6))
# plt.plot(solT1.t, x, color = [0, 0.4470, 0.7410], label = 'X')
# plt.plot(solT1.t, y, color = [0.8500, 0.3250, 0.0980], label = 'Y')
# plt.plot(solT1.t, z, color = [0.9290, 0.6940, 0.1250], label = 'Z')

# plt.plot(solT2.t, x2, color = [0, 0.4470, 0.7410])
# plt.plot(solT2.t, y2, color = [0.8500, 0.3250, 0.0980])
# plt.plot(solT2.t, z2, color = [0.9290, 0.6940, 0.1250])

# plt.plot(solT3.t, x3, color = [0, 0.4470, 0.7410])
# plt.plot(solT3.t, y3, color = [0.8500, 0.3250, 0.0980])
# plt.plot(solT3.t, z3, color = [0.9290, 0.6940, 0.1250])


# plt.xlabel('time [TU]')
# plt.ylabel('position [DU]')
# plt.title('Velocity Checking: T1')

# plt.legend()
# plt.grid()

# plt.show()



# plt.figure(figsize=(12, 6))
# plt.plot(solT1.t, solT1.y[3], color = [0, 0.4470, 0.7410], label = 'vX')
# plt.plot(solT1.t, solT1.y[4], color = [0.8500, 0.3250, 0.0980], label = 'vY')
# plt.plot(solT1.t, solT1.y[5], color = [0.9290, 0.6940, 0.1250], label = 'vZ')
# plt.plot(solT2.t, solT2.y[3], color = [0, 0.4470, 0.7410])
# plt.plot(solT2.t, solT2.y[4], color = [0.8500, 0.3250, 0.0980])
# plt.plot(solT2.t, solT2.y[5], color = [0.9290, 0.6940, 0.1250])
# plt.plot(solT3.t, solT3.y[3], color = [0, 0.4470, 0.7410])
# plt.plot(solT3.t, solT3.y[4], color = [0.8500, 0.3250, 0.0980])
# plt.plot(solT3.t, solT3.y[5], color = [0.9290, 0.6940, 0.1250])


# plt.xlabel('time [TU]')
# plt.ylabel('velocity [DU/TU]')
# plt.title('Velocity Checking: T1')

# plt.legend()
# plt.grid()

# plt.show()




# plt.figure(figsize=(12, 6))
# plt.plot(solT1.t, ax/100, color = [0, 0.4470, 0.7410], label = 'aX')
# plt.plot(solT1.t, ay/100, color = [0.8500, 0.3250, 0.0980], label = 'aY')
# plt.plot(solT1.t, az/100, color = [0.9290, 0.6940, 0.1250], label = 'aZ')

# plt.plot(solT2.t, ax2, color = [0, 0.4470, 0.7410])
# plt.plot(solT2.t, ay2, color = [0.8500, 0.3250, 0.0980])
# plt.plot(solT2.t, az2, color = [0.9290, 0.6940, 0.1250])

# plt.plot(solT3.t, ax3, color = [0, 0.4470, 0.7410])
# plt.plot(solT3.t, ay3, color = [0.8500, 0.3250, 0.0980])
# plt.plot(solT3.t, az3, color = [0.9290, 0.6940, 0.1250])


# plt.xlabel('time [TU]')
# plt.ylabel('accel [DU/TU^2]')
# plt.title('Velocity Checking: T1')

# plt.legend()
# plt.grid()

# plt.show()