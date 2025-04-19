# Peter Lie
# AERO 500 / 470

# ICA 4/5

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()


# 1. Identify a simple (one abstraction away) base/derived class 
# relationship

print("Problem 1: ")
# Base Class
class Satellite:
    agency = "NASA"
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def orbit(self):
        return "Earth SOI"

# Derived Class
class LOP(Satellite):
    def __init__(self, name, type):
        super().__init__(name, type)
        # self.name = name
        # self.type = type

    def orbit(self):
        return "Lunar Orbit!" # Abstraction

print("Base Class: ")
my_Sat = Satellite("Generic Starlink", "Comms Sat")
print("Name: " , my_Sat.name)
print("Type: " , my_Sat.type)
print("Orbit:" , my_Sat.orbit())

print("Derived Class: ")
Gateway = LOP("Gateway", "Space Station")
print("Name: " , Gateway.name)
print("Type: " , Gateway.type)
print("Orbit:" , Gateway.orbit())
print("The derived class LOP is one abstraction away from the base class Satellite with the orbit regime")


# 2. Define 1-2 attributes and 1-2 methods of the base class, 
# mix class and object attributes

# Attribute describes, method acts

print(" ")
print("Problem 2: ")

# Using the same base Class
class Satellite:
    # Class attribute
    agency = "NASA"

    # Constructor with object attributes
    def __init__(self, name, type):
        self.name = name       # Object attribute
        self.type = type       # Object attribute

    # Instance method
    def orbit(self):
        return f"{self.name} is orbiting within Earth's Sphere of Influence."

    # Another instance method
    def payload(self):
        return f"{self.name} carries a payload."

# Example usage
Hubble = Satellite("Hubble", "Telescope")
print(Hubble.orbit())
print(Hubble.payload())
print(f"{Hubble.name} belongs to {Hubble.agency}.")


# 3. Develop 2-3 derived classes from the base class

print(" ")
print("Problem 3:")

# LOP from Part 1
class LOP(Satellite):
    pass
    # def __init__(self, name, type):
    #     super().__init__(name, type)


print("Derived Class 1: ")
print("Name: " , Gateway.name)
print("Type: " , Gateway.type)
print(f"{Gateway.name} belongs to {Gateway.agency}.")
print("Orbit:" , Gateway.orbit())
print("The derived class LOP is one abstraction away from the base class Satellite with the orbit regime")


# Shuttle:
class Shuttle(Satellite):
    def __init__(self, name, type):
        super().__init__(name, type)
    def orbit(self):
        return "Low Earth Orbit" # Abstraction
    

print("Derived Class 2: ")
Discovery = Shuttle("Discovery", "Space Shuttle")
print("Name: " , Discovery.name)
print("Type: " , Discovery.type)
print(f"{Discovery.name} belongs to {Discovery.agency}.")
print("Orbit:" , Discovery.orbit())
print("The derived class Shuttle is one abstraction away from the base class Satellite with the orbit regime")



# 4. Override some of the base class attributes and methods in the derived classes

print(" ")
print("Problem 4:")

# LOP from Part 1
class LOP(Satellite):
    agency = "Multinational" # Overwritten

    # Constructor with object attributes
    def __init__(self, name, type):
        self.name = name       # Object attribute
        self.type = type       # Object attribute

    # Overwrite method
    def orbit(self):
        return f"{self.name} is orbiting within Moon's Sphere of Influence."

    # Overwriten another method
    def payload(self):
        return f"{self.name} carries many payloads into cislunar space."

    # def __init__(self, name, type):
    #     super().__init__(name, type)


print("Derived Class 1: ")
Gateway = LOP("Gateway", "Space Station")
print("Name: " , Gateway.name)
print("Type: " , Gateway.type)
print(f"{Gateway.name} belongs to {Gateway.agency}.")
print("Orbit:" , Gateway.orbit())
# print("The derived class LOP is one abstraction away from the base class Satellite with the orbit regime")



# 5. Add a few methods to the derived classes

print(" ")
print("Problem 5:")

# Shuttle:
class Shuttle(Satellite):
    def __init__(self, name, type):
        super().__init__(name, type)
    def orbit(self):
        return "Low Earth Orbit" # Abstraction
    def repair(self):
        return f"{self.name} can repair satellites in orbit."
    def landing(self):
        return f"{self.name} glides back to the Earth."

print("Derived Class 2: ")
Discovery = Shuttle("Discovery", "Space Shuttle")
print("Name:   " , Discovery.name)
print(f"{Discovery.name} belongs to {Discovery.agency}.")
print("Orbit:  " , Discovery.orbit())
print("Repair: " , Discovery.repair())
print("Landing:" , Discovery.landing())
print("The derived class Shuttle is three methods away from the base class Satellite")


# 6. Override foundational methods like __str__ (required) and some others like, __add__, __eq__, etc.

print(" ")
print("Problem 6:")

# Shuttle:
class Shuttle(Satellite):
    def __init__(self, name, type):
        super().__init__(name, type)
    def __str__(self):
        return f"{self.name} is a {self.type}"
    def __add__(self,x,y):
        return f"{self.name} has {x+y} components"
    def orbit(self):
        return "Low Earth Orbit" # Abstraction
    def repair(self):
        return f"{self.name} can repair satellites in orbit."
    def landing(self):
        return f"{self.name} glides back to the Earth."

print("Derived Class 2: ")
Discovery = Shuttle("Discovery", "Space Shuttle")
print("Name:   " , Discovery.name)
print(Discovery.__str__())
print(Discovery.__add__(2,4))
print(f"{Discovery.name} belongs to {Discovery.agency}.")
print("Orbit:  " , Discovery.orbit())
print("Repair: " , Discovery.repair())
print("Landing:" , Discovery.landing())
print("The derived class Shuttle is three methods away from the base class Satellite")


# 7. Create a module from your class definitions and create a new python file to show how your classes work.

# See File ICA5

