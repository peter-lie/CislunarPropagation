# Peter Lie
# AERO 500 / 470

# ICA 3

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()


# 1. Using a for loop, write Python code to compute the factorial of the integer n.

n = 6
factorial = 1

for i in range(1,n+1):
    factorial *= i
    i += 1

print("Problem 1: ")
print(n , "factorial is:" , factorial)


# 2. Use nested for loops to compute the product of two matrices.  Use a list of lists to represent the matrix data (rows and columns).

print(" ")
print("Problem 2: ")

matrix1R1 = [1, 2, 3]
matrix1R2 = [4, 5, 6]
matrix1R3 = [7, 8, 9]

matrix1 = [matrix1R1, matrix1R2, matrix1R3]
print("Matrix 1:" , matrix1)

matrix2R1 = [9, 8, 7]
matrix2R2 = [6, 5, 4]
matrix2R3 = [3, 2, 1]

matrix2 = [matrix2R1, matrix2R2, matrix2R3]
print("Matrix 2:" , matrix2)

rows = len(matrix1)
rowlength = len(matrix1[0])

# print("Test:" , matrix2C1[0])

MatrixProduct = [[0 for w in range(rows)] for w in range(rowlength)]
for i in range(rows):
    for j in range(rowlength):
        for k in range(rows):
            MatrixProduct[i][j] += matrix1[i][k] * matrix2[k][j]

print("MatrixProduct:" , MatrixProduct)



# 3. Given the two lists below, use list comprehension to find the product 
# of the elements of the list.  i.e., in Matlab like code, find 
# mass.*value = [1.7*2, 4.2*3, 2.6*(-1), 5.4*5].

# Hint:  Use the zip() function.

mass  = [1.7, 4.2, 2.6, 5.4]
value = [2, 3, -1, 5]

zipVal = zip(mass, value)
product = [m * v for m, v in zipVal]
# Dunno what is up with this floating point issue

print(" ")
print("Problem 3: ")

print(product)

