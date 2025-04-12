# Peter Lie
# AERO 500 / 470


# ICA 1

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()


# 1. Define a variable x and set it equal to 5
x = 5
print('1. x =', x)


# 2a. Define a variable l as an empty list
l = []

# 2b. Add five numbers to your list
l.extend([1, 2, 3, 5, 8])
print('2. l =', l)


# 3. Use list comprehension to multiply each element of l by x and print the result
xl = [i * x for i in l]
print('3. xl =', xl)


# 4a. Define a dictionary D that maps any five 'dec' values of the ASCII codes to the appropriate printable character https://en.wikipedia.org/wiki/ASCII#Printable_charactersLinks to an external site..
D = {65: 'A', 69: 'E', 80: 'P', 82: 'R',  84: 'T'}
# 4b. Insert the key/value pair for the "!" and the number "2".
D[33] = '!'
D[50] = '2'
print('4. D =', D)


# 5. Sort the new dictionary and print each key/value pair with the format, "The decimal number, 65, is the ASCII code for the character 'A'"  Hint: Use the sorted() function and dictionary comprehension.

Dsort = {k: D[k] for k in sorted(D)}
print('5. D =', Dsort)

for k, v in Dsort.items():
    print(f"The decimal number, {k}, is the ASCII code for the character '{v}'")


