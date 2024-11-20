# Peter Lie
# AERO 599
# Orbit optimization in Python

import os
clear = lambda: os.system('clear')
clear()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.optimize import root
import matplotlib.pyplot as plt

# Constants
Ve = 0.9  # Exhaust velocity
T = 0.1   # Maximum thrust

# Initial conditions
x0, y0 = 1.05, 0
vx0, vy0 = 0, 1.05 ** (-0.5)
m0 = 1
tspan = [0, 6]  # Initial guess for the time span
tol = 1.5e-10       # Tolerance for ODE and optimization

init_state = [x0, y0, vx0, vy0, m0, -1, 0, 0, -1, -1]  # Initial state with adjoint variables

def burn_coast_burn(t, state, T, Ve):
    x, y, vx, vy, m, λ1, λ2, λ3, λ4, λ5 = state
    r = np.sqrt(x**2 + y**2)

    ## MATLAB
    λv = np.sqrt(λ3**2 + λ4**2)
    ux = -λ3 / λv
    uy = -λ4 / λv
    # Control
    c3 = T

    # Equations of motion
    x_dot = vx
    y_dot = vy
    vx_dot = -(x / r**3) + (T / m) * ux
    vy_dot = -(y / r**3) + (T / m) * uy
    m_dot = - T / Ve

    # From Hst
    λ1_dot = λ3*(1/(x**2 + y**2)**(3/2) - (3*x**2)/(x**2 + y**2)**(5/2)) - (3*λ4*x*y)/(x**2 + y**2)**(5/2)
    λ2_dot = λ4*(1/(x**2 + y**2)**(3/2) - (3*y**2)/(x**2 + y**2)**(5/2)) - (3*λ3*x*y)/(x**2 + y**2)**(5/2)
    λ3_dot = -λ1
    λ4_dot = -λ2
    λ5_dot = - (c3*λ3**2)/(m**2*λv) - (c3*λ4**2)/(m**2*λv)

    return [x_dot, y_dot, vx_dot, vy_dot, m_dot, λ1_dot, λ2_dot, λ3_dot, λ4_dot, λ5_dot]


# Solve initial guess trajectory
sol = solve_ivp(burn_coast_burn, tspan, init_state, args=(T, Ve), rtol=tol, atol=tol)

# Extract results
t = sol.t
x, y, vx, vy, m, λ1, λ2, λ3, λ4, λ5 = sol.y

switching_function = np.sqrt(λ3**2 + λ4**2) * Ve + λ5 * m

# Plot trajectory
theta = np.linspace(0, 2 * np.pi, 100)
plt.figure()
plt.plot(x, y, label='Continuous Transfer Orbit')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.grid()
plt.title('Initial Propagation')
plt.show()

# Plot switching function
plt.figure()
plt.plot(t, switching_function, label='Switching Function')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Switching Function')
plt.grid()
plt.title('Switching Function')
plt.show()


# Use Fsolve to get to final position with continuous burn
final_time_guess = 6

fsolve_init_state = np.hstack([init_state, final_time_guess])

def burn_fsolve(fsolve_init_state):
    state_init = fsolve_init_state[0:10]
    tspan_end = fsolve_init_state[10]

    Ve = .9
    T = .1
    tspan = [0, tspan_end]

    sol = solve_ivp(burn_coast_burn, tspan, state_init, args=(T, Ve), method='RK45')
    t = sol.t
    x, y, vx, vy, m, λ1, λ2, λ3, λ4, λ5 = sol.y

    s0 = [x[0], y[0], vx[0], vy[0], m[0]]
    # s0 = sol.y[0:5,1] # Initial state
    sf = [x[-1], y[-1], vx[-1], vy[-1], m[-1]]
    # sf = sol.y[0:5,-1] # Final state

    # lambda0 = stateout1(1,6:10)
    lambdaf = [λ1[-1], λ2[-1], λ3[-1], λ4[-1], λ5[-1]]

    # lambdav = sqrt(lambdaf(3)**2 + lambdaf(4)**2)

    # cf1 = -lambdaf(3) / lambdav
    # cf2 = -lambdaf(4) / lambdav
    # c3 = T

    rf = np.sqrt(sf[0]**2 + sf[1]**2)
    vf = np.sqrt(sf[2]**2 + sf[3]**2)

    # Forcing
    F0 = rf - 2; # Force final radius to be 2
    F1 = vf - np.sqrt(1/2); # Force final velocity to be a**(-1/2)
    F2 = sf[0]*sf[3] - sf[1]*sf[2] - np.sqrt(2); # And in the right direction as well

    F3 = lambdaf[4] + 1; # Force lambda5 to -1

    # Dependency equation
    F4 = -lambdaf[0]* (sf[1]/sf[0]) + lambdaf[1] - lambdaf[2]* (sf[3]/sf[2]) * ((sf[2] + sf[1]*sf[3]/sf[0])/(sf[0] + sf[3]*sf[1]/sf[2])) + lambdaf[3] * ((sf[2] + sf[1]*sf[3]/sf[0])/(sf[0] + sf[3]*sf[1]/sf[2]))

    # H equation (extra NBC)
    F5 = lambdaf[0] * sf[2] + lambdaf[1] * sf[3] + lambdaf[2] * (-sf[0]/rf**3) + lambdaf[3] * ((-sf[1]/rf**3))
    # F(6,1) = lambdaf(1) * sf(3) + lambdaf(2) * sf(4) + lambdaf(3) * (-sf(1)/2**3) + lambdaf(4) * ((-sf(2)/2**3));

    # Starting conditions
    F6 = s0[0] - 1.05 # Starting x position
    F7 = s0[1] # Starting y position 0
    F8 = s0[2] # Starting x velocity is 0 (circular)
    F9 = s0[3] - (1.05**(-.5)) # Starting y velocity is a**(-1/2)
    F10 = s0[4] - 1 # Starting mass

    return [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]


optimized_init_burn = fsolve(burn_fsolve, fsolve_init_state)

# Pull out initial guess conditions and final time

optimized_init = optimized_init_burn[0:10] # Q2.S0 = Q2.X(1:10)
tspan1 = optimized_init_burn[10] # Q2.tspan = [0 Q2.X(11)]

t_span = [0, tspan1]

# Propagate with optimized initial conditions
sol2 = solve_ivp(burn_coast_burn, t_span, optimized_init, args=(T, Ve), rtol=tol, atol=tol)

# Extract results
t = sol2.t
x, y, vx, vy, m, λ1, λ2, λ3, λ4, λ5 = sol2.y

switching_function = np.sqrt(λ3**2 + λ4**2) * Ve + λ5 * m

# Plot trajectory
theta = np.linspace(0, 2 * np.pi, 100)
plt.figure()
plt.plot(x, y, label='Continuous Transfer Orbit')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Optimized Propagation')
plt.show()

# Plot switching function
plt.figure()
plt.plot(t, switching_function, label='Switching Function')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Switching Function')
plt.grid()
plt.title('Switching Function')
plt.show()


# guess = [2.6, 4, 6]  # Initial time guesses
# optimal_times = fsolve(burn_fsolve, guess)

tspan3 = [0, 2]
tspan4 = [2, t[-1]]
tspan5 = [t[-1], 1.2*t[-1]]

# Burn
sol3 = solve_ivp(burn_coast_burn, tspan3, optimized_init, args=(T, Ve), rtol=tol, atol=tol)
new_initial = sol3.y[:,-1]

# Coast
sol4 = solve_ivp(burn_coast_burn, tspan4, new_initial, args=(0, Ve), rtol=tol, atol=tol)
new_initial = sol4.y[:,-1]

# Burn
sol5 = solve_ivp(burn_coast_burn, tspan5, new_initial, args=(T, Ve), rtol=tol, atol=tol)

# Combine results
t_combined = np.hstack((sol3.t, sol4.t, sol5.t))
state_combined = np.hstack((sol3.y, sol4.y, sol5.y))

x,y,vx,vy,m,λ1,λ2,λ3,λ4,λ5= state_combined[0,:],state_combined[1,:],state_combined[2,:],state_combined[3,:],state_combined[4,:],state_combined[5,:],state_combined[6,:],state_combined[7,:],state_combined[8,:],state_combined[9,:], 

switching_function = np.sqrt(λ3**2 + λ4**2) * Ve + λ5 * m

# Plot trajectory
theta = np.linspace(0, 2 * np.pi, 100)
plt.figure()
plt.plot(x, y, label='Burn Coast Burn Transfer Orbit')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Initial Propagation')
plt.show()

# Plot switching function
plt.figure()
plt.plot(t_combined, switching_function, label='Switching Function')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Switching Function')
plt.grid()
plt.title('Switching Function')
plt.show()

# Burn Coast Burn Fsolve
# BCB_fsolve_init = np.hstack([optimized_init, 2, t[-1], 1.2*t[-1], 0, 0])
BCB_fsolve_init = np.hstack([optimized_init, 1, 3.6, 1, 0, 0])


def BCB_fsolve(fsolve_init_state):
    state_init = fsolve_init_state[0:10]

    t1 = np.abs(fsolve_init_state[10])
    t2 = np.sum(np.abs(fsolve_init_state[11]) + np.abs(fsolve_init_state[10]))
    tf = np.sum(np.abs(fsolve_init_state[12])+ np.abs(fsolve_init_state[11]) + np.abs(fsolve_init_state[10]))

    print("Initial: ", state_init, t1, t2, tf)

    Ve = .9
    T = .1
    tspan1 = [0, t1]
    tspan2 = [t1, t2]
    tspan3 = [t2, tf]

    sol3 = solve_ivp(burn_coast_burn, tspan1, state_init, args=(T, Ve), rtol=tol, atol=tol)
    x3, y3, vx3, vy3, m3, λ1, λ2, λ3_3, λ4_3, λ5_3 = sol3.y
    new_initial = sol3.y[:,-1]

    # Coast
    sol4 = solve_ivp(burn_coast_burn, tspan2, new_initial, args=(0, Ve), rtol=tol, atol=tol)
    x4, y4, vx4, vy4, m4, λ1, λ2, λ3_4, λ4_4, λ5_4 = sol4.y
    new_initial = sol4.y[:,-1]

    # Burn
    sol5 = solve_ivp(burn_coast_burn, tspan3, new_initial, args=(T, Ve), rtol=tol, atol=tol)
    x5, y5, vx5, vy5, m5, λ1_5, λ2_5, λ3_5, λ4_5, λ5_5 = sol5.y

    # Initial and final states
    sf = sol5.y[0:5,-1]

    # lambda0 = stateout1(1,6:10)
    lambdaf = sol5.y[5:10,-1]

    # Switching function
    switching1 = np.sqrt(λ3_3**2 + λ4_3**2) * Ve + λ5_3 * m3
    switching2 = np.sqrt(λ3_4**2 + λ4_4**2) * Ve + λ5_4 * m4
    switching3 = np.sqrt(λ3_5**2 + λ4_5**2) * Ve + λ5_5 * m5
    
    rf = np.sqrt(sf[0]**2 + sf[1]**2)
    # vf = np.sqrt(sf[2]**2 + sf[3]**2)

    # Forcing
    F0 = rf - 2; # Force final radius to be 2
    F1 = sf[2]**2 + sf[3]**2 - 1/2; # Force final velocity to be a**(-1/2)
    F2 = sf[0]*sf[3] - sf[1]*sf[2] - np.sqrt(2); # And in the right direction as well

    F3 = lambdaf[4] + 1; # Force lambda5 to -1

    # Dependency equation
    F4 = -lambdaf[0]* (sf[1]/sf[0]) + lambdaf[1] - lambdaf[2]* (sf[3]/sf[2]) * ((sf[2] + sf[1]*sf[3]/sf[0])/(sf[0] + sf[3]*sf[1]/sf[2])) + lambdaf[3] * ((sf[2] + sf[1]*sf[3]/sf[0])/(sf[0] + sf[3]*sf[1]/sf[2]))

    # H equation (extra NBC)
    F5 = lambdaf[0] * sf[2] + lambdaf[1] * sf[3] + lambdaf[2] * (-sf[0]/rf**3) + lambdaf[3] * ((-sf[1]/rf**3))
    # F(6,1) = lambdaf(1) * sf(3) + lambdaf(2) * sf(4) + lambdaf(3) * (-sf(1)/2**3) + lambdaf(4) * ((-sf(2)/2**3));

    # Starting conditions
    F6 = x3[0] - 1.05 # Starting x position
    F7 = y3[0] # Starting y position 0
    F8 = vx3[0] # Starting x velocity is 0 (circular)
    F9 = vy3[0] - (1.05**(-.5)) # Starting y velocity is a**(-1/2)
    F10 = m3[0] - 1 # Starting mass is 1

    # Switching Function Zeros at on/off
    F11 = switching1[0]
    F12 = switching1[-1]
    F13 = switching2[-1]
    F14 = switching3[-1]

    return [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14]


# np.scipy.optimize.root
optimized_init_BCB = fsolve(BCB_fsolve, BCB_fsolve_init)

print("optimized_init_BCB:" , optimized_init_BCB)
print("optimized_init_BCB09:" , optimized_init_BCB[0:10])
print("optimized_init_BCB10:" , optimized_init_BCB[10])
print("optimized_init_BCB11:" , optimized_init_BCB[11])
print("optimized_init_BCB12:" , optimized_init_BCB[12])

BCB_optimized_state = optimized_init_BCB[0:10]
optimized_t1 = optimized_init_BCB[10]
optimized_t2 = np.sum(optimized_init_BCB[11] + optimized_init_BCB[10])
optimized_tf = np.sum(optimized_init_BCB[12] + optimized_init_BCB[11] + optimized_init_BCB[10])

tspan3 = [0, optimized_t1]
tspan4 = [optimized_t1, optimized_t2]
tspan5 = [optimized_t2, optimized_tf]

# Burn
sol3 = solve_ivp(burn_coast_burn, tspan3, BCB_optimized_state, args=(T, Ve), rtol=tol, atol=tol)
x3, y3, vx3, vy3, m3, λ1_3, λ2_3, λ3_3, λ4_3, λ5_3 = sol3.y
new_initial = sol3.y[:,-1]

# Coast
sol4 = solve_ivp(burn_coast_burn, tspan4, new_initial, args=(0, Ve), rtol=tol, atol=tol)
x4, y4, vx4, vy4, m4, λ1_4, λ2_4, λ3_4, λ4_4, λ5_4 = sol4.y
new_initial = sol4.y[:,-1]

# Burn
sol5 = solve_ivp(burn_coast_burn, tspan5, new_initial, args=(T, Ve), rtol=tol, atol=tol)
x5, y5, vx5, vy5, m5, λ1_5, λ2_5, λ3_5, λ4_5, λ5_5 = sol5.y

# Combine results
t_combined = np.hstack((sol3.t, sol4.t, sol5.t))
state_combined = np.hstack((sol3.y, sol4.y, sol5.y))

x,y,vx,vy,m,λ1,λ2,λ3,λ4,λ5= state_combined[0,:],state_combined[1,:],state_combined[2,:],state_combined[3,:],state_combined[4,:],state_combined[5,:],state_combined[6,:],state_combined[7,:],state_combined[8,:],state_combined[9,:], 

switching_function = (λ3**2 + λ4**2)**.5 * Ve + λ5 * m

switching_function1 = (λ3_3**2 + λ4_3**2)**.5 * Ve + λ5_3 * m3
switching_function2 = (λ3_4**2 + λ4_4**2)**.5 * Ve + λ5_4 * m4
switching_function3 = (λ3_5**2 + λ4_5**2)**.5 * Ve + λ5_5 * m5



# Plot trajectory
theta = np.linspace(0, 2 * np.pi, 100)
plt.figure()
plt.plot(x, y, label='Burn Coast Burn Transfer Orbit')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Optimized Propagation')
plt.show()


# Trajectory Broken Up
plt.figure()
plt.plot(x3, y3, label='Burn 1', color = 'red')
plt.plot(x4, y4, label='Coast', color = 'green')
plt.plot(x5, y5, label='Burn 2', color = 'purple')
plt.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), label='Initial Orbit')
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label='Final Orbit')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Optimized Propagation')
plt.show()



# Plot switching function
plt.figure()
plt.plot(t_combined, switching_function, label='Switching Function')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Switching Function')
plt.grid()
plt.title('Switching Function')
plt.show()


# Plot switching function
plt.figure()
plt.plot(sol3.t, switching_function1, label='Burn 1', color = 'red')
plt.plot(sol4.t, switching_function2, label='Coast', color = 'green')
plt.plot(sol5.t, switching_function3, label='Burn 2', color = 'purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc = 'upper right')
plt.grid()
plt.title('Switching Function')
plt.show()


# Mass History Plot
plt.figure()
plt.plot(t_combined, m, label='Mass History')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Mass')
plt.grid()
plt.title('Mass History')
plt.show()


