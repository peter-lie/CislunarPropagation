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

# state1 is 31600 km DRO:
state1 = [9.0668606911016691E-1,	2.1794721283701204E-27,	-9.7234137635134883E-35,	-4.5013918683951635E-16,	4.8268635279185140E-1,	-2.3891922680462994E-36] # 1.1478169491263817E+0 TU period (5.08 days) Stability index:	1.0000000000000000E+0
time1 = 1.1478 # TU

# state2 is 38700 km DRO:
state2 = [8.8851644651130812E-1,	-2.1123241028442816E-27,	-2.4052359292251004E-35,	3.5284399524651394E-14,	4.7144916062570680E-1,	-8.7954528090354503E-36] # 1.5040078008734745E+0 TU period (6.67 days) Stability index:	1.0000000000000000E+0
time2 = 1.5

# state3 is 46000 km DRO:
state3 = [8.6974743065322846E-1,	4.4729783197777055E-28,	-1.0076006967462480E-35,	-1.8932888296551354E-14,	4.7098755399886844E-1,	1.2233222772213693E-35] # 1.8864822791798597E+0 TU period (8.36 days) Stability index:	1.0000000000021900E+0
time3 = 1.9

# state4 is 53400 km DRO:
state4 = [8.5090059903659454E-1,	-2.9582039642797011E-29,	2.5713077966461058E-37,	5.9291764738006365E-14,	4.7854916068880948E-1,	4.3892775388711900E-37] # 2.2785533531021711E+0 TU period (10.10 days) Stability index:	1.0000000000000000E+0
time4 = 2.3

# state5 is 60600 km DRO:
state5 = [8.3243496520463345E-1,	9.6283409466007841E-24,	-4.9805377233953735E-2,	-9.2128417379751029E-13,	4.9189498557003636E-1,	2.8882629226696031E-25] # 2.6632503069191058E+0 TU period (11.81 days) Stability index:	1.0000000002987699E+0
time5 = 2.7

# state6 is 70000km DRO
state6 = [8.0591079311650515E-1,	2.1618091280991729E-23,	3.4136631163268282E-25,	-8.1806482539864240E-13,	5.1916995982435687E-1,	-5.7262098359472236E-25] # 3.2014543457713667E+0 TU period (14.25 days) Stability index:	1.0000000001631200E+0
time6 = 3.2

# state7 is 80400 km DRO:
state7 = [7.8155463946457815E-1,	1.2893005469358983E-24,	2.4477492849665138E-25,	-5.4593904644267237E-13,	5.5107030763312925E-1,	1.5622142851225408E-26] # 3.6639841308759213E+0 TU period (16.24 days) Stability index:	1.0000000000810700E+0
time7 = 3.7

# state8 is 90800 km DRO:
state8 = [7.5492481353799945E-1,	-1.4407278756733903E-23,	2.5951594942468979E-25,	-5.4236902956776887E-13,	5.9215966279011512E-1,	-3.0032480436606385E-25] # 4.1187678443895797E+0 TU period (18.26 days) Stability index:	1.0000000000386700E+0
time8 = 4.2

# state9 is 101200 km DRO:
state9 = [7.2826047814297090E-1,	2.2862170922332342E-23,	1.2446498309080938E-24,	-3.7725883661223612E-13,	6.3882904855069411E-1,	-2.8114470190595486E-25] # 4.5116876314821610E+0 TU period (19.99 days) Stability index:	1.0000000000057001E+0
time9 = 4.6


DROstate = state8
Timestate = time8

# moon distance in km
moondist = (1 - mu - DROstate[0]) * 384.4e3
# print(moondist): 69937.2 km

# Time span for the propagation 
t_span1 = (0, Timestate)  # Start and end times
t_span2 = (0, 1*2*np.pi) #
# t_eval = np.linspace(0, 29.46, 1000)  # Times to evaluate the solution, use for ECI plotting


# Solve the IVP
sol0_3BPNRHO = solve_ivp(cr3bp_equations, t_span1, state0, args=(mu,), rtol=tol, atol=tol)
sol0_3BPDRO = solve_ivp(cr3bp_equations, t_span1, DROstate, args=(mu,), rtol=tol, atol=tol)

# sol1_4BPNRHO = solve_ivp(bcr4bp_equations, t_span2, state0, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
# sol1_4BPDRO = solve_ivp(bcr4bp_equations, t_span2, state1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)


# Hypothetical transfer maneuvers
# Starting with 3BP NRHO characteristics, looking for 3BP DRO characteristics

# Loop to check for the last time orbit crosses the xy plane inside of the DRO

theta0 = 0
thetastep = np.pi/8
# thetastep = np.pi/256 # 3 hour runtime maybe?
thetamax = 2 * np.pi + thetastep
deltavmin = 1
thetamin = 0
# thrustangle = np.pi/4; # rad, 45 deg .584 with 16 points, .421 with 512 points
# thrustangle = np.pi/3; # rad, 60 deg .584 with 16 points, .460 with 512 points
# thrustangle = np.pi/6; # rad, 30 deg .584 with 16 points, .436 with 512 points
# thrustangle = np.pi/2; # rad, 90 deg .584 with 16 points, .405 with 512 points
# thrustangle = 5*np.pi/12; # rad, 75 deg .584 with 16 points, .415 with 512 points
# thrustangle = np.pi/12; # rad, 15 deg .584 with 16 points, .440 with 512 points
thrustangle = 0; # rad, 0 deg .584 with 16 points, . with 512 points

vyoffset = 0    # 0 gives 0.521 km/s with 32 points
                # -.1 yeilds 0.483 km/s with 64 points
                # negative y benefits the points close to the positive x axis, and allows for better curvature


moondistSQ = (1*(moondist/384.4e3))**2
deltavstorage = {}

while theta0 < thetamax:
    print('theta0: ', theta0)
    tspant1 = (0,14) # for 0 z position
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
    
    # Dr. V suggested trying a 2 body lamberts here to try to target best velocity


    newstate1 = solT1.y[:,-1] + [0, 0, 0, 0, vyoffset, -vzend]
    tspant3 = (tend,tend+3)
    deltav1 = np.sqrt(vyoffset**2 + vzend**2)
    
    solT2 = solve_ivp(bcr4bp_equations, tspant3, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
    x = solT2.y[0,:]
    y = solT2.y[1,:]
    z = solT2.y[2,:]
    vx = solT2.y[3,:]
    vy = solT2.y[4,:]
    vz = solT2.y[5,:]
    t2 = solT2.t

    # Check if trajectory off the end intersects with DRO
    r = []
    for i in range(0,len(x)):
        for j in range(0,len(sol0_3BPDRO.y[0,:])):
            trajectorydistance = np.sqrt((x[i] - sol0_3BPDRO.y[0,j])**2 + (y[i] - sol0_3BPDRO.y[1,j])**2 + (z[i] - sol0_3BPDRO.y[2,j])**2)
            r.append((i, j, trajectorydistance))
    
    cpa = min(r, key=lambda e: e[2])
    i, j, cpavalue = cpa
    checkdistance = 1e-2

    if cpavalue < checkdistance:
        endpoint = (x[i], y[i], z[i])
        endtime = t2[i]
        # print(endtime)
        tspant4 = (tend,endtime)
        solT3 = solve_ivp(bcr4bp_equations, tspant4, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
        
        deltav2 = np.sqrt((-vx[i]+sol0_3BPDRO.y[3,j])**2 + (-vy[i]+sol0_3BPDRO.y[4,j])**2)

        deltav = deltav1 + deltav2
        # print('  deltav1: ', deltav1, 'DU/TU')
        # print('  deltav2: ', deltav2, 'DU/TU')
        DUtokm = 384.4e3 # kms in 1 DU
        TUtoS4 = 406074.761647 # s in 1 4BP TU
        deltavS = deltav * DUtokm / TUtoS4
        print('  deltavS: ', deltavS, 'km/s')
        deltavstorage[theta0] = deltavS


    else:
        while cpavalue > checkdistance:
            moonx = xend - (1-mu)
            moony = yend
            moonangle = np.arctan2(moony,moonx)
            # print('  moonangle:',moonangle)
            xvel += .05*np.cos(moonangle - thrustangle)         # Try different angles here
            yvel += .05*np.sin(moonangle - thrustangle)
            newstate1 = solT1.y[:,-1] + [0, 0, 0, xvel, yvel, -vzend]
            tspant3 = (tend,tend+5)
            
            solT2 = solve_ivp(bcr4bp_equations, tspant3, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
            x = solT2.y[0,:]
            y = solT2.y[1,:]
            z = solT2.y[2,:]
            vx = solT2.y[3,:]
            vy = solT2.y[4,:]
            vz = solT2.y[5,:]
            t2 = solT2.t

            # Check if trajectory off the end intersects with DRO
            r = []
            for i in range(0,len(x)):
                for j in range(0,len(sol0_3BPDRO.y[0,:])):
                    trajectorydistance = np.sqrt((x[i] - sol0_3BPDRO.y[0,j])**2 + (y[i] - sol0_3BPDRO.y[1,j])**2 + (z[i] - sol0_3BPDRO.y[2,j])**2)
                    r.append((i, j, trajectorydistance))

            cpa = min(r, key=lambda e: e[2])
            i, j, cpavalue = cpa


        deltav1 = np.sqrt(xvel**2 + yvel**2 + vzend**2)
        deltav2 = np.sqrt((-vx[i]+sol0_3BPDRO.y[3,j])**2 + (-vy[i]+sol0_3BPDRO.y[4,j])**2)
        deltav = deltav1 + deltav2
        DUtokm = 384.4e3 # kms in 1 DU
        TUtoS4 = 406074.761647 # s in 1 4BP TU
        deltavS = deltav * DUtokm / TUtoS4
        print('  deltavS: ', deltavS, 'km/s')
        deltavstorage[theta0] = deltavS


            # # Plot the trajectory
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # Plot Moon, Lagrange Points
            # ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=30)  # Secondary body (Moon)
            # ax.scatter([L1_x], [0], [0], color=[0.4660, 0.6740, 0.1880], s=15, label='L1')
            # ax.scatter([L2_x], [0], [0], color=[0.3010, 0.7450, 0.9330], s=15, label='L2')

            # # Plot the trajectories
            # ax.plot(sol0_3BPNRHO.y[0], sol0_3BPNRHO.y[1], sol0_3BPNRHO.y[2], color=[0, 0.4470, 0.7410], label='9:2 NRHO')
            # ax.plot(sol0_3BPDRO.y[0], sol0_3BPDRO.y[1], sol0_3BPDRO.y[2], color=[0.4940, 0.1840, 0.5560], label='70000km DRO')

            # ax.plot(solT1.y[0], solT1.y[1], solT1.y[2], color=[0.9290, 0.6940, 0.1250], label='T 1')
            # ax.scatter([newstate1[0]], [newstate1[1]], [newstate1[2]], color=[0.8500, 0.3250, 0.0980], s=10, label='Maneuver')
            # ax.plot(solT2.y[0], solT2.y[1], solT2.y[2], color=[0.4660, 0.6740, 0.1880], label='T 2')

            # # Labels and plot settings
            # ax.set_xlabel('x [DU]')
            # ax.set_ylabel('y [DU]')
            # ax.set_zlabel('z [DU]')
            # ax.set_title(f'Solar Theta0: {theta0}')

            # ax.legend(loc='best')
            # plt.show()

    if deltavS < deltavmin:
        deltavmin = deltavS
        thetamin = theta0

    theta0 += thetastep


# Check length of array to ensure all points found a solution
# print(len(deltavstorage))

# Save to a new file
# with open("ThrustAngle45-16.txt", "w") as file:  # Use "a" instead of "w" to append to the file
#     file.write(output_data)


# import json

# with open("ThrustAngle0-512.json", "w") as file:     # Change filename
#     json.dump(deltavstorage, file)


print('     deltavmin:', deltavmin)
print('     @theta0:', thetamin)


plt.figure(figsize=(10, 6))
plt.plot(deltavstorage.keys(), deltavstorage.values())

plt.xlabel('Solar Theta0 [rads]')
plt.ylabel('deltaV [km/s]')
plt.title('deltaV vs Theta0')

plt.show()



# Reading in data
# with open("ThrustAngle60-16.json", "r") as file:
#     dataplot1 = json.load(file)
# with open("ThrustAngle45-16.json", "r") as file:
#     dataplot2 = json.load(file)


# plt.figure(figsize=(10, 6))
# plt.plot(dataplot1.keys(), dataplot1.values(), color=[0, 0.4470, 0.7410])
# plt.plot(dataplot2.keys(), dataplot2.values(), color=[0.8500, 0.3250, 0.0980])

# plt.xlabel('Solar Theta0 [rads]')
# plt.ylabel('deltaV [km/s]')
# plt.title('deltaV vs Theta0')
# plt.xticks([0, len(dataplot1)/(2*np.pi)-1/6, 2*len(dataplot1)/(2*np.pi)-1/3, 3*len(dataplot1)/(2*np.pi)-1/2, 4*len(dataplot1)/(2*np.pi)-2/3, 5*len(dataplot1)/(2*np.pi)-5/6, 6*len(dataplot1)/(2*np.pi)-1], ['0', '1', '2', '3', '4', '5', '6'])
# plt.show()





# theta0 = 197 * np.pi / 128
theta0 = thetamin

tspant1 = (0,14) # for 0 z position
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
tspant3 = (tend,tend+3)
deltav1 = np.sqrt(vyoffset**2 + vzend**2)

solT2 = solve_ivp(bcr4bp_equations, tspant3, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
x = solT2.y[0,:]
y = solT2.y[1,:]
z = solT2.y[2,:]
vx = solT2.y[3,:]
vy = solT2.y[4,:]
vz = solT2.y[5,:]
t2 = solT2.t

# Check if trajectory off the end intersects with DRO
r = []
for i in range(0,len(x)):
    for j in range(0,len(sol0_3BPDRO.y[0,:])):
        trajectorydistance = np.sqrt((x[i] - sol0_3BPDRO.y[0,j])**2 + (y[i] - sol0_3BPDRO.y[1,j])**2 + (z[i] - sol0_3BPDRO.y[2,j])**2)
        r.append((i, j, trajectorydistance))

cpa = min(r, key=lambda e: e[2])
i, j, cpavalue = cpa
checkdistance = 1e-2

if cpavalue < checkdistance:
    endpoint = (x[i], y[i], z[i])
    endtime = t2[i]
    # print(endtime)
    tspant4 = (tend,endtime)
    solT3 = solve_ivp(bcr4bp_equations, tspant4, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
    
    deltav2 = np.sqrt((-vx[i]+sol0_3BPDRO.y[3,j])**2 + (-vy[i]+sol0_3BPDRO.y[4,j])**2)

    deltav = deltav1 + deltav2
    # print('  deltav1: ', deltav1, 'DU/TU')
    # print('  deltav2: ', deltav2, 'DU/TU')
    DUtokm = 384.4e3 # kms in 1 DU
    TUtoS4 = 406074.761647 # s in 1 4BP TU
    deltavS = deltav * DUtokm / TUtoS4
    print('  deltavS: ', deltavS, 'km/s')
    deltavstorage[theta0] = deltavS
    if deltavS < deltavmin:
        deltavmin = deltavS
        thetamin = theta0

else:
    while cpavalue > checkdistance:
        moonx = xend - (1-mu)
        moony = yend
        moonangle = np.arctan2(moony,moonx)
        print('  moonangle:',moonangle)
        xvel += .05*np.sin(moonangle + np.pi/4)
        yvel += -.05*np.cos(moonangle + np.pi/4)
        newstate1 = solT1.y[:,-1] + [0, 0, 0, xvel, yvel, -vzend]
        tspant3 = (tend,tend+3)
        
        solT2 = solve_ivp(bcr4bp_equations, tspant3, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
        x = solT2.y[0,:]
        y = solT2.y[1,:]
        z = solT2.y[2,:]
        vx = solT2.y[3,:]
        vy = solT2.y[4,:]
        vz = solT2.y[5,:]
        t2 = solT2.t

        # Check if trajectory off the end intersects with DRO
        r = []
        for i in range(0,len(x)):
            for j in range(0,len(sol0_3BPDRO.y[0,:])):
                trajectorydistance = np.sqrt((x[i] - sol0_3BPDRO.y[0,j])**2 + (y[i] - sol0_3BPDRO.y[1,j])**2 + (z[i] - sol0_3BPDRO.y[2,j])**2)
                r.append((i, j, trajectorydistance))

        cpa = min(r, key=lambda e: e[2])
        i, j, cpavalue = cpa

    endtime = t2[i]
    tspant4 = (tend,endtime)
    solT3 = solve_ivp(bcr4bp_equations, tspant4, newstate1, args=(mu,inc,Omega0,theta0,), rtol=tol, atol=tol)
    
    deltav1 = np.sqrt(xvel**2 + yvel**2 + vzend**2)
    deltav2 = np.sqrt((-vx[i]+sol0_3BPDRO.y[3,j])**2 + (-vy[i]+sol0_3BPDRO.y[4,j])**2)
    deltav = deltav1 + deltav2
    DUtokm = 384.4e3 # kms in 1 DU
    TUtoS4 = 406074.761647 # s in 1 4BP TU
    deltavS = deltav * DUtokm / TUtoS4
    print('  deltavS: ', deltavS, 'km/s')
    deltavstorage[theta0] = deltavS





# Plot solT1, solT3

# 3D Plotting

# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Plot the celestial bodies
# # ax.scatter(-mu, 0, 0, color='blue', label='Earth', s=100)  # Primary body (Earth)
ax.scatter(1 - mu, 0, 0, color='gray', label='Moon', s=30)  # Secondary body (Moon)

# # Plot the Lagrange points
ax.scatter([L1_x], [0], [0], color=[0.4660, 0.6740, 0.1880], s=15, label='L1')
ax.scatter([L2_x], [0], [0], color=[0.3010, 0.7450, 0.9330], s=15, label='L2')

# # ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=15, label='Langrage Points')

# # Plot the trajectories
ax.plot(sol0_3BPNRHO.y[0], sol0_3BPNRHO.y[1], sol0_3BPNRHO.y[2], color=[0, 0.4470, 0.7410], label='9:2 NRHO')
ax.plot(sol0_3BPDRO.y[0], sol0_3BPDRO.y[1], sol0_3BPDRO.y[2], color=[0.4940, 0.1840, 0.5560], label='70000km DRO')
# ax.plot(sol0_DROfind.y[0], sol0_DROfind.y[1], sol0_DROfind.y[2], color=[0.4940, 0.1840, 0.5560], label='Target DRO')

# ax.plot(solT0.y[0], solT0.y[1], solT0.y[2], color=[0.9290, 0.6940, 0.1250], label='T 1')
# ax.scatter([newstate2[0]], [newstate2[1]], [newstate2[2]], color=[0.8500, 0.3250, 0.0980], s=10, label='Maneuver')
ax.plot(solT1.y[0], solT1.y[1], solT1.y[2], color=[0.9290, 0.6940, 0.1250], label='Coast Trajectory')

# ax.scatter([newstate3[0]], [newstate3[1]], [newstate3[2]], color=[0.8500, 0.3250, 0.0980], s=10)
# # ax.plot(solT2.y[0], solT2.y[1], solT2.y[2], color=[0, 0.4470, 0.7410], label='T 3')

# # ax.scatter([newstate4[0]], [newstate4[1]], [newstate4[2]], color=[0.8500, 0.3250, 0.0980], s=10)
ax.plot(solT3.y[0], solT3.y[1], solT3.y[2], color=[0.4660, 0.6740, 0.1880], label='DRO Intercept') # [0.9290, 0.6940, 0.1250]

# # ax.plot(sol1_4BPNRHO.y[0], sol1_4BPNRHO.y[1], sol1_4BPNRHO.y[2], color=[0.9290, 0.6940, 0.1250], label='9:2 NRHO')
# # ax.plot(sol1_4BPDRO.y[0], sol1_4BPDRO.y[1], sol1_4BPDRO.y[2], color=[0.4940, 0.1840, 0.5560], label='70000km DRO')


# # Labels and plot settings
ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')

# # ax.set_axis_off()  # Turn off the axes for better visual appeal

ax.legend(loc='best')

# # plt.gca().set_aspect('equal', adjustable='box')
plt.show()



