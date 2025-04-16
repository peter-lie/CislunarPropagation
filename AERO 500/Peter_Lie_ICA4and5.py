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
my_NRHO = LOP("Gateway", "Space Station")
print("Name: " , my_NRHO.name)
print("Type: " , my_NRHO.type)
print("Orbit:" , my_NRHO.orbit())
print("The derived class LOP is one abstraction away from the base class Satellite with the orbit regime")


# 2. Define 1-2 attributes and 1-2 methods of the base class, 
# mix class and object attributes

# Attribute describes, method acts

print(" ")
print("Problem 2: ")

# Using the same base Class
class Satellite:
    # Class attribute (shared by all instances)
    agency = "NASA"

    # Constructor with object attributes
    def __init__(self, name, type):
        self.name = name       # Object attribute
        self.type = type       # Object attribute
        self.objects = []      # Object attribute (list unique to each instance)

    # Instance method
    def orbit(self):
        return f"{self.name} is orbiting within Earth's Sphere of Influence."

    # Add another instance method
    def add_payload(self, payload):
        self.objects.append(payload)
        return f"Payload '{payload}' added to {self.name}."

# Example usage
sat1 = Satellite("Hubble", "Telescope")
print(sat1.orbit())
print(sat1.add_payload("Wide Field Camera"))
print(f"{sat1.name} belongs to {sat1.agency}.")


# 3. Develop 2-3 derived classes from the base class

print(" ")
print("Problem 3:")

# LOP from Part 1
class LOP(Satellite):
    def __init__(self, name, type, payload):
        super().__init__(name, type, payload)

print("Derived Class: ")
# my_NRHO = LOP("Gateway", "Space Station", "Lunar Lander")
print("Name: " , my_NRHO.name)
print("Type: " , my_NRHO.type)
print("Orbit:" , my_NRHO.orbit())
print("The derived class LOP is one abstraction away from the base class Satellite with the orbit regime")


# Shuttle:
class Shuttle(Satellite):
    def __init__(self, name, type, payload):
        super().__init__(name, type, payload)





# 4. Override some of the base class attributes and methods in the derived classes





# 5. Add a few methods to the derived classes





# 6. Override foundational methods like __str__ (required) and some others like, __add__, __eq__, etc.





# 7. Create a module from your class definitions and create a new python file to show how your classes work.





