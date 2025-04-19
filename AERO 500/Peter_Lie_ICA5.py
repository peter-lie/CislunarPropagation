# Peter Lie
# AERO 500 / 470

# ICA 4/5

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()


import Peter_Lie_ICA4 as ICA4

# All commands run and print the same outputs

# Base Class
Satellite = ICA4.Satellite

# Derived Classes
LOP = ICA4.LOP
Shuttle = ICA4.Shuttle

print(" ")
print("Test: ")

Discovery = Shuttle("Discovery", "STS-41")
print("Mission: " , Discovery.type)

