# Peter Lie
# AERO 500 / 470

# ICA 4/5

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()


# 1. Identify a simple (one abstraction away) base/derived class relationship

print("Problem 1: ")
# Base Class
class Satellite:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def orbit(self):
        return "Low Earth Orbit"

class LOP(Satellite):
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def orbit(self):
        return "Lunar Orbit!"

print("Base Class: ")
my_Sat = Satellite("Generic Starlink", "Comms Sat")
print("Name: " , my_Sat.name)
print("Type: " , my_Sat.type)
print("Orbit:" , my_Sat.orbit())

print("Derived Class: ")
my_NRHO = LOP("Gateway", "Space Station")
print("Name: " , my_NRHO.name)
print("Type: " , my_NRHO.type)
print("Orbit:" , my_NRHO.orbit())


# 2. Define 1-2 attributes and 1-2 methods of the base class, mix class and object attributes

print(" ")
print("Problem 2: ")



# 3. Develop 2-3 derived classes from the base class


# 4. Override some of the base class attributes and methods in the derived classes


# 5. Add a few methods to the derived classes


# 6. Override foundational methods like __str__ (required) and some others like, __add__, __eq__, etc.


# 7. Create a module from your class definitions and create a new python file to show how your classes work.

