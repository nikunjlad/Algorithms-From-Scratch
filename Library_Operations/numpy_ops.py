"""
Basic Numpy Operations

NOTE: Uncomment each section to understand the output of it
"""

import numpy as np

# ----- 1. Seeding and Random number generation -----
# seed = 42
# rng = np.random.default_rng(seed)   # creates a random number generator with given seed value

# # loop 5 times to show how generator produces 10 numbers randomly
# # Output is:
# # Pass #0: Random Nums: [5 6 0 7 3 2 4 9 1 8]
# # Pass #1: Random Nums: [4 8 2 6 5 9 7 3 0 1]
# # Pass #2: Random Nums: [0 9 8 4 7 2 3 5 6 1]
# # Pass #3: Random Nums: [6 1 2 7 9 5 8 4 0 3]
# # Pass #4: Random Nums: [3 2 6 5 1 4 8 9 0 7]
# # Running this script any number of times always produces these same 5 combinations of 10 numbers
# # this is governed by the seed value. If seed is changed, the combination will change
# # As long as the seed stays constant, the permutations will be same across runs
# for i in range(5):
#     random_nums = rng.permutation(10)
#     print(f"Pass #{i}: Random Nums: {random_nums}")
# ------------------------------------------------------

# # ----- 2. Random 1D array initialization -----
seed = 42
rng = np.random.default_rng(seed)   # random number generator

# # method 1 - Random 1D float array of 5 elements
# Random function always gives values between 0-1 as uniform distribution
# arr_1d_float = rng.random(5)
# print(f"Random 1D float array: {arr_1d_float}")

# # method 2 - Random 1D float array 5 elements with low-high bounds
# uniform function gives values between low and high values as a uniform distribution
# arr_1d_float_bounds = rng.uniform(low=1.0,high=10.0,size=5)
# print(f"Random 1D float array bounded: {arr_1d_float_bounds}")

# # method 3 - Random 1D integer array of 5 elements
# arr_1d_integer = rng.integers(5, size=5)
# print(f"Random 1D integer array: {arr_1d_integer}")

# # method 4 - Random 1D intger array of 5 elements with low-high bounds
# arr_1d_integer_bounds = rng.integers(low=1,high=10,size=5)
# print(f"Random 1D integer bounded array: {arr_1d_integer_bounds}")
# # -------------------------------------------------------

# # ----- 3. Random 2D array initialization -----
# seed = 42
# rng = np.random.default_rng(seed)   # random number generator

# # method 1 - Random 2D float array of (5,3) elements
# # Random function always gives values between 0-1 as uniform distribution
# arr_2d_float = rng.random((5,3))
# print(f"Random 1D float array: \n {arr_2d_float}")

# # method 2 - Random 2D float array (5,3) elements with low-high bounds
# # uniform function gives values between low and high values as a uniform distribution
# arr_2d_float_bounds = rng.uniform(low=1.0,high=10.0,size=(5,3))
# print(f"Random 2D float array bounded: \n {arr_2d_float_bounds}")

# # method 3 - Random 2D integer array of (5,3) elements
# # integers gives values in range 0 to any user provided high value.
# # below example, the 5 is high=5 with implicit low=0
# arr_2d_integer = rng.integers(5, size=(5,3))
# print(f"Random 2D integer array: \n {arr_2d_integer}")

# # method 4 - Random 2D intger array of (5,3) elements with low-high bounds
# # providing low and high to integers bounds it
# arr_2d_integer_bounds = rng.integers(low=1,high=10,size=(5,3))
# print(f"Random 2D integer bounded array: \n {arr_2d_integer_bounds}")
# -------------------------------------------------------


rng.random


