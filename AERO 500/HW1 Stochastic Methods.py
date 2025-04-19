# Peter Lie
# AERO 500 / 470

# Homework 1: Stochastic Methods

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()

# Task 1:
# Use the Monte Carlo Integration Method to estimate the value of pi
# Submit your Python code and a plot of your results
# Approximately how many iterations does your code take to converge to a solution?
# How many trials did your simulation need to converge within 1% of the known value of pi?

import random
import matplotlib.pyplot as plt
import numpy as np

# Parameters
num_points = 1000
inside_circle = 0
pi_estimates = []

# Monte Carlo simulation
for i in range(1, num_points + 1):
    x, y = random.random(), random.random()  # Point in unit square
    if x**2 + y**2 <= 1:
        inside_circle += 1
    pi_estimate = 4 * inside_circle / i
    pi_estimates.append(pi_estimate)

# Plotting the convergence
plt.figure(figsize=(10, 6))
plt.plot(pi_estimates, label='Estimated ${\\pi}$')
plt.axhline(y=np.pi, color='r', linestyle='--', label='Actual ${\\pi}$')
plt.xlabel('Number of Points')
plt.ylabel('Estimated ${\\pi}$')
plt.title('Monte Carlo Estimation of ${\\pi}$')
plt.legend()
plt.grid(True)
plt.show()

# Convergence checks
within_1_percent = [i for i, est in enumerate(pi_estimates) if abs(est - np.pi) / np.pi < 0.01]
if within_1_percent:
    print(f" estimate converged within 1% after approximately {within_1_percent[0]} iterations.")
else:
    print("π estimate did not converge within 1% in given trials.")

print(f"Final estimate of π after {num_points} iterations: {pi_estimates[-1]}")

