"""
optimization
Core module for trajectory optimization

Functions are very specific to each problem, parameters will generally need 
to be changed for each specific application

"""

# AERO 557 Advanced Orbital Mechanics
# Functions for orbit optimization in Python
# As each function needs to be created specifically for the problem statement, 
# included functions match up with certain homework problems


import numpy as np
from scipy.integrate import solve_ivp
# from scipy.optimize import fsolve
# import matplotlib.pyplot as plt


"""
AERO 557 Homework 3 Problem 3

"""
# Two dimensional motion to end up at y = 1 while maximizing x velocity

# Define the equations of motion:
# Differential equations for 2D motion
# Need as many adjoints as variables in the state

def steering_eq(t, state):
    """
    Adopted from Dr. Abercromby's lectures, equations of motion given
    Use with integrator to propagate in time

    Args:
        time [characteristic time]
        state variables (x and y position and velocity) [characteristic dimensions]
        costate variables [dimensionless]
    
    Returns:
        state derivatives (x and y velocity and acceleration) [characteristic dimensions]
        costate derivatives [dimensionless]

    """

    x, y, xdot, ydot, lambda1, lambda2, lambda3, lambda4 = state
  
    lambdav = np.sqrt(lambda3**2 + lambda4**2)

    # Equations of motion
    xddot = - lambda3 / lambdav
    yddot = - lambda4 / lambdav

    # Adjoint equations
    lambda1_dot = 0
    lambda2_dot = 0
    lambda3_dot = -lambda1
    lambda4_dot = -lambda2
    
    return [xdot, ydot, xddot, yddot, lambda1_dot, lambda2_dot, lambda3_dot, lambda4_dot]


# Fsolve equations:
def steering_eq_fsolve(init_state, t_span):
    """
    Adopted from Dr. Abercromby's lectures
    Equivalent to MATLAB Fsolve
    Varies input arguments to get Fsolves constraints to equal zero

    Args:
        time span [characteristic time]
        initial state guess:
            initial state variables (x and y position and velocity) [characteristic dimensions]
            initial costate variables [dimensionless]
    
    Returns:
        function constraints
            these are all of things that the Fsolve attempts to set to zero 
            to apply the proper constraints onto the trajectory
            can include forcing initial or final position or velocity components
            NOT output in the function call, can display to terminal (workspace)
        optimized initial state and costate variables
            the required initial state and costate varibles that satisfy the 
            differential equations and Fsolve function

    """


    # Default tolerance for ODE and optimization
    tol = 1e-9

    # Propagate the state with the given initial conditions
    sol = solve_ivp(steering_eq, t_span, init_state, rtol=tol, atol=tol)
    final_state = sol.y[:, -1]  # Final state

    # Fsolve constraints: 
    # These equations are what Fsolve sends to 0 to comply with the state and adjoint requirements
    # Final constraints
    final_y = final_state[1] - 1  # Final height = 1
    final_ydot = final_state[3]   # Final vertical velocity = 0
    final_lambda1 = final_state[4] # Final lambda1 = 0
    final_lambda3 = final_state[6] + 1 # Final lambda3 = -1
    # Initial constraints
    initial_state1 = init_state[0]  # Initial positions and velocities are 0
    initial_state2 = init_state[1]  # Initial positions and velocities are 0
    initial_state3 = init_state[2]  # Initial positions and velocities are 0
    initial_state4 = init_state[3]  # Initial positions and velocities are 0

    return [final_y, final_ydot, final_lambda1, final_lambda3, initial_state1, initial_state2, initial_state3, initial_state4]


"""
AERO 557 Homework 4 Problem 2
Optimize 2D transfer trajectory between circular orbits for lowest delta-V (highest 
ending mass) in polar coordinates for a burn coast burn (BCB) thrust profile with free time

"""


def burn_coast_burn(t, state, T, Ve):
    """
    Adopted from Dr. Abercromby's lectures, equations of motion given
    Use with integrator to propagate in time

    Args:
        time t [characteristic time]
        state:
            state variables (x and y position and velocity, mass) [characteristic dimensions]
            costate variables [dimensionless]
        thrust T [nondimensional force]
        exit velocity Ve [nondimensional speed]

    
    Returns:
        state derivatives (x and y velocity and acceleration, mass derivative) [characteristic dimensions]
        costate derivatives [dimensionless]

    """

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



def burn_fsolve(fsolve_init_state):
    """
    Adopted from Dr. Abercromby's lectures, equations of motion given
    Use with integrator to propagate in time
    First optimization using a continuous burn to approximate the behavior of a BCB
    and solve for switching function
    Trajectory will not be perfect as it arrives at destination orbit before initial guess for final time

    Args:
        initial state:
            state variables (x and y position and velocity, mass) [characteristic dimensions]
            costate variables [dimensionless]
            guess for final time

    Returns:
        function constraints
            see previous description
            NOT output in the function call, can display to terminal (workspace)
        optimized initial state and costate variables
            the required initial state and costate varibles that satisfy the 
            differential equations and Fsolve function

    """

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


def BCB_fsolve(fsolve_init_state):
    """
    Adopted from Dr. Abercromby's lectures, equations of motion given
    Use with integrator to propagate in time
    Performs same optimization on a burn coast burn (BCB) thrust profile

    Args:
        initial state:
            state variables (x and y position and velocity, mass) [characteristic dimensions]
            costate variables [dimensionless]
            guess for time spans [characeristic time]
                first burn time span
                coast time span
                second burn time span
                dummy variables (zeroes) to get the number of Fsolve arguments to match

    Returns:
        function constraints
            see previous description
            NOT output in the function call, can display to terminal (workspace)
        optimized initial state and costate variables
            the required initial state and costate varibles that satisfy the 
            differential equations and Fsolve function

    """


    tol = 1e-9

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