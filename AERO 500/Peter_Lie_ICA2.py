# Peter Lie
# AERO 500 / 470

# ICA 2

# Clear terminal
import os
clear = lambda: os.system('clear')
clear()


# 1. Using a while loop, write Python code to compute the factorial of the integer n

n = 4
factorial = 1

while n > 0:
    factorial *= n 
    n -= 1
print("Problem 1: ")
print(factorial)


# 2. Use a while loop, with an if statement to print out the even numbers from 0 to 20.  In this activity, use the modulus operator (%)

print(" ")
print("Problem 2: ")

i = 0

while i < 21:
    rem = i%2 
    if rem == 0:
        print(i)
    i += 1


# 3. Use an if/else statement and the ‘random’ module (https://www.w3schools.com/python/gloss_python_random_number.aspLinks to an external site.) to code up the rock-paper-scissors game.  Your code should ask the user to pick between rock, paper, or scissors, then generate a random choice from the computer.  The user can enter ‘exit’ to quit the game.

import random

print(" ")
print("Problem 3: ")

while True:
    n = input("Enter 1 for rock, 2 for paper, or 3 for scissors, or Exit for exit: ")
    # n = 1 (rock)
    # n = 2 (paper)
    # n = 3 (scissors)

    if n == 'Exit':
        break
    else:
        n = int(n)
        c = random.randrange(1,4)
        # c = 1 (rock)
        # c = 2 (paper)
        # c = 3 (scissors)

        if c == n:
            # Same symbol, tie
            print("It's a tie!")
        elif c == n-1:
            # user beats computer
            print("You win!")
        elif c == n+2:
            # user beats computer
            print("You win!")

        elif c == n+1:
            # computer beats user
            print("You lose!")
        elif c == n-2:
            # computer beats user
            print("You lose!")

        # else:
            # pass



    

