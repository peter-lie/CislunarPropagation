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
theta0 = np.pi/2 # true anomaly of sun at start
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


# Solar acceleration as a function of position
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


# Solar position over time
r_Sx0, r_Sy0, r_Sz0 = sun_position(0, inc, Omega0, theta0)
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

# [1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07] zimovan spreen
# Initial State Vectors
# state0 is 9:2 NRHO
state0 = [1.0213448959167291E+0,	-4.6715051049863432E-27,	-1.8162633785360355E-1,	-2.3333471915735886E-13,	-1.0177771593237860E-1,	-3.4990116102675334E-12]
state1 = [1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07]
state2 = [1.0311112407855418E+0,	-7.1682537158522033E-28,	-1.8772287481683392E-1,	-6.0418753714167314E-15,	-1.2222545055005253E-1,	-2.3013319140654479E-13]
state3 = [1.0410413863429402E+0,	2.5382291997301062E-27,     -1.9270725388017840E-1,	-1.3345031213261042E-14,	-1.4112595315877385E-1,	-3.8518451969245009E-13]	
state4 = [1.0510246802643324E+0,	-4.9400712148877512E-27,	-1.9671629109317987E-1,	5.2659007647464425E-15,	    -1.5829051257176247E-1,	-3.7699709298926106E-15]	
state5 = [1.0608510547453884E+0,	1.4576285469039374E-27, 	-1.9970484051930923E-1,	1.2985946081601830E-14,	    -1.7342665527688911E-1,	1.2992733483228851E-13]
state6 = [1.0703566401901397E+0,	5.5809992744091920E-28, 	-2.0160273066948275E-1,	1.0006785921793148E-14,	    -1.8640880976946453E-1,	-1.4975544143063520E-14]	
state7 = [1.0795687536743368E+0,	-3.7070501190356569E-27,	-2.0235241192732428E-1,	9.8926663976217253E-15,	    -1.9739555890728064E-1,	2.3849287871166338E-15]	
state8 = [1.0885078929557530E+0,	2.6633614957150261E-27,	    -2.0185192176893893E-1, 1.0402854351281896E-14,	    -2.0648787544408306E-1,	1.9564110320959405E-14]
state9 = [1.0969915644292789E+0,	-1.3906893609387349E-27,	-2.0004092398988046E-1,	4.9047891808896165E-15,	    -2.1360265558047956E-1,	-6.6909069485186488E-15]
state10 = [1.1051670425311451E+0,	-8.1953151848621478E-27,	-1.9685536017603811E-1,	1.6943363216580394E-15,	    -2.1897531097302853E-1,	-5.9229263353083240E-15]
state11 = [1.1131543850560104E+0,	2.5445211165619221E-27,	    -1.9219343403238076E-1,	-9.2916156853283232E-16,	-2.2273474373483881E-1,	9.2655794365851240E-15]	
state12 = [1.1208633587786683E+0,	-1.8419747099550102E-27,	-1.8609585622736360E-1,	1.3903995116060766E-15,	    -2.2489246199372176E-1,	8.7488992941048550E-15]	
state13 = [1.1283622742910746E+0,	2.3645660744182675E-27,	    -1.7854427340541623E-1,	-4.5798028934849037E-16,	-2.2553137627959235E-1,	2.5413136885628749E-15]
state14 = [1.1379400821512216E+0,	1.1284777304996799E-27,	    -1.6639035702326449E-1,	3.7117943876736724E-15,	    -2.2409655788940144E-1,	4.1749877139974635E-15]	
state15 = [1.1524322574193573E+0,	1.8730227951073470E-27,	    -1.4170575255213977E-1,	1.9359598525413841E-15,	    -2.1629186612466694E-1,	2.9604326939793333E-15]
state16 = [1.1639036554477815E+0,	9.1296013915460989E-28,	    -1.1429073408508866E-1,	6.7733850720026752E-16,	    -2.0341018387035803E-1,	-2.3459051588153798E-15]
state17 = [1.1722585986699103E+0,	-1.3565689601167387E-28,	-8.5283282230768045E-2,	-8.9502376867247273E-16,	-1.8748883050291434E-1,	-3.5915603942748612E-15]	
state18 = [1.1776634616218469E+0,   9.1332746632727835E-28,	    -5.4782557212334590E-2,	-1.9915282629756328E-15,	-1.7111373252726161E-1,	-1.3631030574735472E-14]	
state19 = [1.1804065710733480E+0,	-3.4636147407686238E-27,	-2.2265439866267177E-2,	4.2499992018699737E-15,	    -1.5866687154938475E-1,	-2.1371460332864722E-15]	
state20 = [1.1808985038227133E+0,	6.5447955993483541E-28,	    2.4054798067734207E-4,	2.2497506560355833E-15,	    -1.5585658836610825E-1,	1.7974530092170653E-17]	
state21 = [1.1808065620052182E+0,	-1.2663416563930241E-27,	-9.7067024054411886E-3,	2.9845318416276489E-15,	    -1.5640080244193136E-1,	4.8069583851150394E-17]

# Time span for the propagation
t_span = (0, 4.4)  # Start and end times

# Solve the system of equations
sol0 = solve_ivp(bcr4bp_equations, t_span, state0, args=(mu, inc, Omega0, theta0,), rtol=tol, atol=tol)
sol2 = solve_ivp(bcr4bp_equations, t_span, state2, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol4 = solve_ivp(bcr4bp_equations, t_span, state4, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol6 = solve_ivp(bcr4bp_equations, t_span, state6, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol8 = solve_ivp(bcr4bp_equations, t_span, state8, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol10 = solve_ivp(bcr4bp_equations, t_span, state10, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol12 = solve_ivp(bcr4bp_equations, t_span, state12, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol14 = solve_ivp(bcr4bp_equations, t_span, state14, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol15 = solve_ivp(bcr4bp_equations, t_span, state15, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol16 = solve_ivp(bcr4bp_equations, t_span, state16, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol17 = solve_ivp(bcr4bp_equations, t_span, state17, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol18 = solve_ivp(bcr4bp_equations, t_span, state18, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol19 = solve_ivp(bcr4bp_equations, t_span, state19, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol20 = solve_ivp(bcr4bp_equations, t_span, state20, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)
sol21 = solve_ivp(bcr4bp_equations, t_span, state21, args=(mu,inc, Omega0, theta0,), rtol=tol, atol=tol)


# 3D Plotting

# Plot the trajectory
fig = plt.figure()
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
# ax.plot_surface(xearth, yearth, zearth, color=cearth, alpha=0.8, linewidth=0)

# Plot the Lagrange points
# ax.scatter([L2_x], [0], [0], color='red', s=15, label='L2')
# ax.scatter([L1_x, L2_x, L3_x, L4_x, L5_x], [0, 0, 0, L4_y, L5_y], [0, 0, 0, 0, 0], color='red', s=15, label='Langrage Points')

# Plot the trajectory of the small object
ax.plot(sol0.y[0], sol0.y[1], sol0.y[2], color='orange', label='9:2 NRHO')
# ax.plot(sol2.y[0], sol2.y[1], sol2.y[2], color=(0,0,128/255), label='Trajectory')
# ax.plot(sol4.y[0], sol4.y[1], sol4.y[2], color=(0,15/255,128/255))
# ax.plot(sol6.y[0], sol6.y[1], sol6.y[2], color=(0,30/255,128/255))
# ax.plot(sol8.y[0], sol8.y[1], sol8.y[2], color=(0,45/255,128/255))
# ax.plot(sol10.y[0], sol10.y[1], sol10.y[2], color=(0,60/255,128/255))
# ax.plot(sol12.y[0], sol12.y[1], sol12.y[2], color=(0,75/255,128/255))
# ax.plot(sol14.y[0], sol14.y[1], sol14.y[2], color=(0,90/255,128/255))
# ax.plot(sol15.y[0], sol15.y[1], sol15.y[2], color=(0,100/255,128/255))
# ax.plot(sol16.y[0], sol16.y[1], sol16.y[2], color=(0,110/255,128/255))
# ax.plot(sol17.y[0], sol17.y[1], sol17.y[2], color=(0,120/255,128/255))
# ax.plot(sol18.y[0], sol18.y[1], sol18.y[2], color=(0,130/255,128/255))
# ax.plot(sol19.y[0], sol19.y[1], sol19.y[2], color=(0,140/255,128/255))
# ax.plot(sol20.y[0], sol20.y[1], sol20.y[2], color=(0,150/255,128/255))
# ax.plot(sol21.y[0], sol21.y[1], sol21.y[2], color=(0,160/255,128/255))


# ax.scatter(r_Sx0 /200 , r_Sy0 /200, r_Sz0 /200, color='yellow', s=80, label='Sun') # Sun at starting position
# ax.scatter(r_Sx1 /200 , r_Sy1 /200, r_Sz1 /200, color='yellow', s=80)
# ax.scatter(r_Sx2 /200 , r_Sy2 /200, r_Sz2 /200, color='yellow', s=80)
# ax.scatter(r_Sx3 /200 , r_Sy3 /200, r_Sz3 /200, color='yellow', s=80)
# ax.text(r_Sx0 /200 -.4 , r_Sy0 /200 - .2, r_Sz0 /200, 'Sun @ t = 0')
# ax.text(r_Sx1 /200 -.4 , r_Sy1 /200 + .2, r_Sz1 /200, 'Sun @ t = pi/2 TU')
# ax.text(r_Sx2 /200 -.15 , r_Sy2 /200 - .2, r_Sz2 /200, 'Sun @ t = pi TU')
# ax.text(r_Sx3 /200 -.4 , r_Sy3 /200 - .2, r_Sz3 /200, 'Sun @ t = 3pi/2 TU')


# Labels and plot settings
ax.set_xlabel('x [DU]')
ax.set_ylabel('y [DU]')
ax.set_zlabel('z [DU]')

xticks = .97, 1, 1.03, 1.06
ax.set_xticks(xticks)

yticks = -.04, 0, .04
ax.set_yticks(yticks)

zticks = -.15, -.1, -.05, 0, .05
ax.set_zticks(zticks)

ax.view_init(elev=15, azim=-20)

# ax.set_title('CR3BP Propagation')
# ax.legend()
# ax.set_box_aspect([2,.4,1]) 
plt.gca().set_aspect('equal', adjustable='box')
plt.show()



# Create subplots
# fig, axes = plt.subplots(1, 3, figsize=(9, 5))
# fig.suptitle("CR3BP: L2S 9:2 NRHO")

# # Top-down view (XY plane)
# axes[0].plot(sol0.y[0], sol0.y[1], color = 'orange', label="9:2 NRHO")
# axes[0].plot(sol2.y[0], sol2.y[1], color=(0,0,128/255), label="Trajectory")
# axes[0].plot(sol4.y[0], sol4.y[1], color=(0,15/255,128/255))
# axes[0].plot(sol6.y[0], sol6.y[1], color=(0,30/255,128/255))
# axes[0].plot(sol8.y[0], sol8.y[1], color=(0,45/255,128/255))
# axes[0].plot(sol10.y[0], sol10.y[1], color=(0,60/255,128/255))
# axes[0].plot(sol12.y[0], sol12.y[1], color=(0,75/255,128/255))
# axes[0].plot(sol14.y[0], sol14.y[1], color=(0,90/255,128/255))
# axes[0].plot(sol15.y[0], sol15.y[1], color=(0,100/255,128/255))
# axes[0].plot(sol16.y[0], sol16.y[1], color=(0,110/255,128/255))
# axes[0].plot(sol17.y[0], sol17.y[1], color=(0,120/255,128/255))
# axes[0].plot(sol18.y[0], sol18.y[1], color=(0,130/255,128/255))
# axes[0].plot(sol19.y[0], sol19.y[1], color=(0,140/255,128/255))
# axes[0].plot(sol20.y[0], sol20.y[1], color=(0,150/255,128/255))
# axes[0].plot(sol21.y[0], sol21.y[1], color=(0,160/255,128/255))
# axes[0].scatter([1 - mu], [0], s=30, color='gray', label="Moon")
# axes[0].scatter([L2_x], [0], color='red', s=20, label='L2')
# axes[0].set_title("Top-down view (XY plane)")
# axes[0].set_xlabel("x [DU]")
# axes[0].set_ylabel("y [DU]")
# axes[0].grid(True)
# axes[0].legend()

# # Side view (YZ plane)
# axes[1].plot(sol0.y[1], sol0.y[2], color = 'orange', label="9:2 NRHO")
# axes[1].plot(sol2.y[1], sol2.y[2], color=(0,0,128/255), label="Trajectory")
# axes[1].plot(sol4.y[1], sol4.y[2], color=(0,15/255,128/255))
# axes[1].plot(sol6.y[1], sol6.y[2], color=(0,30/255,128/255))
# axes[1].plot(sol8.y[1], sol8.y[2], color=(0,45/255,128/255))
# axes[1].plot(sol10.y[1], sol10.y[2], color=(0,60/255,128/255))
# axes[1].plot(sol12.y[1], sol12.y[2], color=(0,75/255,128/255))
# axes[1].plot(sol14.y[1], sol14.y[2], color=(0,90/255,128/255))
# axes[1].plot(sol15.y[1], sol15.y[2], color=(0,100/255,128/255))
# axes[1].plot(sol16.y[1], sol16.y[2], color=(0,110/255,128/255))
# axes[1].plot(sol17.y[1], sol17.y[2], color=(0,120/255,128/255))
# axes[1].plot(sol18.y[1], sol18.y[2], color=(0,130/255,128/255))
# axes[1].plot(sol19.y[1], sol19.y[2], color=(0,140/255,128/255))
# axes[1].plot(sol20.y[1], sol20.y[2], color=(0,150/255,128/255))
# axes[1].plot(sol21.y[1], sol21.y[2], color=(0,160/255,128/255))
# axes[1].scatter([0], [0], s=30, color='gray', label="Moon")
# axes[1].scatter([0], [0], color='red', s=20, label='L2')
# axes[1].set_title("Side view (YZ plane)")
# axes[1].set_xlabel("y [DU]")
# axes[1].set_ylabel("z [DU]")
# axes[1].grid(True)
# axes[1].legend()

# # Moon axis view (XZ plane)
# axes[2].plot(sol0.y[0], sol0.y[2], color = 'orange', label="9:2 NRHO")
# axes[2].plot(sol2.y[0], sol2.y[2], color=(0,0,128/255), label="Trajectory")
# axes[2].plot(sol4.y[0], sol4.y[2], color=(0,15/255,128/255))
# axes[2].plot(sol6.y[0], sol6.y[2], color=(0,30/255,128/255))
# axes[2].plot(sol8.y[0], sol8.y[2], color=(0,45/255,128/255))
# axes[2].plot(sol10.y[0], sol10.y[2], color=(0,60/255,128/255))
# axes[2].plot(sol12.y[0], sol12.y[2], color=(0,75/255,128/255))
# axes[2].plot(sol14.y[0], sol14.y[2], color=(0,90/255,128/255))
# axes[2].plot(sol15.y[0], sol15.y[2], color=(0,100/255,128/255))
# axes[2].plot(sol16.y[0], sol16.y[2], color=(0,110/255,128/255))
# axes[2].plot(sol17.y[0], sol17.y[2], color=(0,120/255,128/255))
# axes[2].plot(sol18.y[0], sol18.y[2], color=(0,130/255,128/255))
# axes[2].plot(sol19.y[0], sol19.y[2], color=(0,140/255,128/255))
# axes[2].plot(sol20.y[0], sol20.y[2], color=(0,150/255,128/255))
# axes[2].plot(sol21.y[0], sol21.y[2], color=(0,160/255,128/255))
# axes[2].scatter([1 - mu], [0], s=30, color='gray', label="Moon")
# axes[2].scatter([L2_x], [0], color='red', s=20, label='L2')
# axes[2].set_title("Side view (XZ plane)")
# axes[2].set_xlabel("x [DU]")
# axes[2].set_ylabel("z [DU]")
# axes[2].grid(True)
# axes[2].legend()

# # Adjust layout and show
# plt.tight_layout()
# plt.show()

