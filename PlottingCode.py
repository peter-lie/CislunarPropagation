# Peter Lie
# AERO 599
# Bicircular Restricted 4 Body Problem
# Earth moon system with solar perturbation, varying starting sun true anomaly

import os
clear = lambda: os.system('clear')
clear()

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

import plotly.express as px
import plotly.graph_objects as go


# Reading in data
# Instantaneous Burn Data
with open("ThrustAngle45-512.json", "r") as file:
    IBdataplot1 = json.load(file)
with open("ThrustAngle60-512.json", "r") as file:
    IBdataplot2 = json.load(file)
with open("ThrustAngle30-512.json", "r") as file:
    IBdataplot3 = json.load(file)
with open("ThrustAngle90-512.json", "r") as file:
    IBdataplot4 = json.load(file)
with open("ThrustAngle75-512.json", "r") as file:
    IBdataplot5 = json.load(file)
with open("ThrustAngle15-512.json", "r") as file:
    IBdataplot6 = json.load(file)
with open("ThrustAngle0-512.json", "r") as file:
    IBdataplot7 = json.load(file)



# Solar Sail Data

with open("SailAm-.001.json", "r") as file:
    SSdataplot00 = json.load(file)
with open("SailAm-.005.json", "r") as file:
    SSdataplot0 = json.load(file)
with open("SailAm-.01.json", "r") as file:
    SSdataplot1 = json.load(file)
with open("SailAm-.05.json", "r") as file:
    SSdataplot2 = json.load(file)
with open("SailAm-.1.json", "r") as file:
    SSdataplot3 = json.load(file)
with open("SailAm-.5.json", "r") as file:
    SSdataplot4 = json.load(file)
with open("SailAm-1.json", "r") as file:
    SSdataplot5 = json.load(file)
with open("SailAm-2.json", "r") as file:
    SSdataplot6 = json.load(file)
with open("SailAm-5.json", "r") as file:
    SSdataplot7 = json.load(file)
with open("SailAm-10.json", "r") as file:
    SSdataplot8 = json.load(file)
with open("SailAm-20.json", "r") as file:
    SSdataplot9 = json.load(file)



# Continuous Thrust Data

with open("ContinuousThrustAC.json", "r") as file:
    CTdataplot1 = json.load(file)
with open("ContinuousThrustVC.json", "r") as file:
    CTdataplot2 = json.load(file)
with open("ContinuousThrustCoC.json", "r") as file:
    CTdataplot3 = json.load(file)
with open("ContinuousThrustVC2.json", "r") as file:
    CTdataplot4 = json.load(file)
with open("ContinuousThrustCoC.5.json", "r") as file:
    CTdataplot5 = json.load(file)
with open("ContinuousThrustCoC2.json", "r") as file:
    CTdataplot6 = json.load(file)
with open("ContinuousThrustVC.5.json", "r") as file:
    CTdataplot7 = json.load(file)






# theta  = 4.835107443415557
# deltaV = 0.405143838212779


# MATLAB Default colors
# 1: [0, 0.4470, 0.7410]        Blue
# 2: [0.8500, 0.3250, 0.0980]   Red
# 3: [0.9290, 0.6940, 0.1250]   Yellow
# 4: [0.4940, 0.1840, 0.5560]   Purple
# 5: [0.4660, 0.6740, 0.1880]   Green?
# 6: [0.3010, 0.7450, 0.9330]
# 7: [0.6350, 0.0780, 0.1840]

alpha1 = 1


plt.figure(figsize=(10, 6))

# Instantaneous Burn Data
# plt.plot(IBdataplot7.keys(), IBdataplot7.values(), color=[0.6350, 0.0780, 0.1840], label='0 Degrees', alpha = alpha1)
# plt.plot(IBdataplot6.keys(), IBdataplot6.values(), color=[0.3010, 0.7450, 0.9330], label='15 Degrees', alpha = alpha1)
# plt.plot(IBdataplot3.keys(), IBdataplot3.values(), color=[0.4660, 0.6740, 0.1880], label='30 Degrees', alpha = alpha1)
# plt.plot(IBdataplot1.keys(), IBdataplot1.values(), color=[0.4940, 0.1840, 0.5560], label='45 Degrees', alpha = alpha1)
plt.plot(IBdataplot2.keys(), IBdataplot2.values(), color=[0.9290, 0.6940, 0.1250], label='60 Degrees', alpha = alpha1)
# plt.plot(IBdataplot5.keys(), IBdataplot5.values(), color=[0.8500, 0.3250, 0.0980], label='75 Degrees', alpha = alpha1)
# plt.plot(IBdataplot4.keys(), IBdataplot4.values(), color=[0, 0.4470, 0.7410] , label='90 Degrees', alpha = alpha1)

# Vertical Lines for Demonstration
# plt.plot((x1, x2), (y1, y2), 'k-')
plt.axvline(x = 217, ymin = 0, ymax = 2, color=[0, 0, 0], alpha = .8) #, **kwargs)
plt.axvline(x = 259, ymin = 0, ymax = 2, color=[0, 0, 0], alpha = .8) #, **kwargs)
plt.axvline(x = 472, ymin = 0, ymax = 2, color=[0, 0, 0], alpha = .8) #, **kwargs)
plt.axvline(x = 512, ymin = 0, ymax = 2, color=[0, 0, 0], alpha = .8) #, **kwargs)

plt.axvline(x = 97, ymin = 0, ymax = 2, color=[0, 0, 0], alpha = .8) #, **kwargs)
plt.axvline(x = 113, ymin = 0, ymax = 2, color=[0, 0, 0], alpha = .8) #, **kwargs)
plt.axvline(x = 353, ymin = 0, ymax = 2, color=[0, 0, 0], alpha = .8) #, **kwargs)
plt.axvline(x = 369, ymin = 0, ymax = 2, color=[0, 0, 0], alpha = .8) #, **kwargs)


# Intersections
plt.plot(217, .603, "x", color='green', alpha = 1) #, **kwargs)
plt.plot(259, .607, "x", color='green', alpha = 1) #, **kwargs)
# plt.plot(472, .607, "x", color=[0.8500, 0.3250, 0.0980], alpha = 1) #, **kwargs)
# plt.plot(512, .596, "x", color=[0.8500, 0.3250, 0.0980], alpha = 1) #, **kwargs)

# plt.plot(97, .916, "x", color=[0.8500, 0.3250, 0.0980], alpha = 1) #, **kwargs)
# plt.plot(113, .925, "x", color=[0.8500, 0.3250, 0.0980], alpha = 1) #, **kwargs)
# plt.plot(353, .911, "x", color=[0.8500, 0.3250, 0.0980], alpha = 1) #, **kwargs)
# plt.plot(369, .918, "x", color=[0.8500, 0.3250, 0.0980], alpha = 1) #, **kwargs)



# Shading
plt.axvspan(217, 259, color='blue', alpha=0.2)
plt.axvspan(472, 512, color='blue', alpha=0.2)

plt.axvspan(97, 113, color='red', alpha=0.2)
plt.axvspan(353, 369, color='red', alpha=0.2)


# For some reason, must keep 1 set of data from instantaneous burns to properly display data
# plt.plot(IBdataplot1.keys(), IBdataplot1.values(), color=[0.4940, 0.1840, 0.5560], alpha = 0.0) # , label='Instantaneous Burns')

# Solar Sail Data
# plt.plot(SSdataplot9.keys(), SSdataplot9.values(), color=[0.9290, 0.6940, 0.1250], label='${\\frac{A}{m} = 20 \: \\frac{m^2}{kg}}$', alpha = 0.8)
# plt.plot(SSdataplot8.keys(), SSdataplot8.values(), color=[0.8500, 0.3250, 0.0980], label='${\\frac{A}{m} = 10 \: \\frac{m^2}{kg}}$', alpha = 0.8)
# plt.plot(SSdataplot7.keys(), SSdataplot7.values(), color=[0.6350, 0.0780, 0.1840], label='${\\frac{A}{m} = 5 \: \\frac{m^2}{kg}}$')
# plt.plot(SSdataplot6.keys(), SSdataplot6.values(), color=[0.3010, 0.7450, 0.9330], label='${\\frac{A}{m} = 2 \: \\frac{m^2}{kg}}$')
# plt.plot(SSdataplot5.keys(), SSdataplot5.values(), color=[0, 0.4470, 0.7410], label='${\\frac{A}{m} = 1 \: \: \\frac{m^2}{kg}}$', alpha = 0.8)
# plt.plot(SSdataplot4.keys(), SSdataplot4.values(), color=[0.9290, 0.6940, 0.1250], label='${\\frac{A}{m} = .5 \: \\frac{m^2}{kg}}$', alpha = 0.9)
# plt.plot(SSdataplot3.keys(), SSdataplot3.values(), color=[0.9290, 0.6940, 0.1250], label='${\\frac{A}{m} = .1 \: \\frac{m^2}{kg}}$', alpha = 0.8)
# plt.plot(SSdataplot2.keys(), SSdataplot2.values(), color=[0.8500, 0.3250, 0.0980], label='${\\frac{A}{m} = .05 \: \\frac{m^2}{kg}}$', alpha = 0.9)
# plt.plot(SSdataplot1.keys(), SSdataplot1.values(), color=[0, 0.4470, 0.7410], label='${\\frac{A}{m} = .01 \: \\frac{m^2}{kg}}$', alpha = 0.7)
# plt.plot(SSdataplot0.keys(), SSdataplot0.values(), color=[0.8500, 0.3250, 0.0980], label='${\\frac{A}{m} = .005 \: \\frac{m^2}{kg}}$', alpha = 0.9)
# plt.plot(SSdataplot00.keys(), SSdataplot00.values(), color=[0, 0.4470, 0.7410], label='${\\frac{A}{m} = .001 \: \\frac{m^2}{kg}}$', alpha = 0.7)


# Continuous Thrust Data
# plt.plot(CTdataplot2.keys(), CTdataplot2.values(), color=[0.8500, 0.3250, 0.0980], label='Continuous Thrust: ${\\vec{V}}$', alpha = 0.9)
# plt.scatter(CTdataplot1.keys(), CTdataplot1.values(), marker = ".", color=[0, 0.4470, 0.7410], label='Continuous Thrust: $-{\\vec{V}}$', alpha = 0.99)
# plt.plot(CTdataplot3.keys(), CTdataplot3.values(), color=[0.4660, 0.6740, 0.1880], label='Continuous Thrust: Control 1', alpha = 0.99)
# plt.plot(CTdataplot4.keys(), CTdataplot4.values(), color=[0.9290, 0.6940, 0.1250], label='Continuous Thrust: $2{\\vec{V}}$', alpha = 0.9)
# plt.plot(CTdataplot5.keys(), CTdataplot5.values(), color=[0, 0.4470, 0.7410], label='Continuous Thrust: Control 3', alpha = 0.9)
# plt.scatter(CTdataplot6.keys(), CTdataplot6.values(), marker = ".", color=[0.4940, 0.1840, 0.5560], label='Continuous Thrust: Control 2', alpha = 0.99)
# plt.plot(CTdataplot7.keys(), CTdataplot7.values(), color=[0.4940, 0.1840, 0.5560], label='Continuous Thrust: ${\\frac{1}{2}\\vec{V}}$', alpha = 0.9)



plt.xlabel(u'Solar ${\\theta_0}$ [rads]')
# plt.xlabel(u'Length ${\mu}m$')

plt.ylabel(u'${\Delta}$v [$\\frac{km}{s}$]')
# plt.title('deltaV vs Theta0')

plt.xticks([0, len(IBdataplot1)/8, len(IBdataplot1)/4, 3*len(IBdataplot1)/8, len(IBdataplot1)/2, 5*len(IBdataplot1)/8, 3*len(IBdataplot1)/4, 7*len(IBdataplot1)/8, len(IBdataplot1)], ['0', '${\\frac{\pi}{4}}$', '${\\frac{\pi}{2}}$', '${\\frac{3\pi}{4}}$', '${\pi}$', '${\\frac{5\pi}{4}}$', '${\\frac{3\pi}{2}}$', '${\\frac{7\pi}{4}}$', '2${\pi}$'])

# plt.xlim(150,320)

plt.yticks([0, .4, .8, 1.2, 1.6, 2])

# plt.ylim(.4,1.4)

plt.legend(loc = 'upper right')
plt.grid()
plt.show()




# # Create Scatter Plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=SSdataplot5.keys(), y=SSdataplot5.values(),
#                     mode='lines', name='lines'))
# fig.add_trace(go.Scatter(x=SSdataplot6.keys(), y=SSdataplot6.values(),
#                     mode='lines', name='lines'))
# fig.add_trace(go.Scatter(x=SSdataplot3.keys(), y=SSdataplot3.values(),
#                     mode='lines', name='lines'))

# fig.show()


