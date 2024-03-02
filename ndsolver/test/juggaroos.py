import matplotlib.pyplot as plt
import numpy as np


# Step 1: Create a grid of x, y, and z values
x = np.linspace(-2, 2, 100)  # Adjust the range and density as needed
y = np.linspace(-1, 1, 100)
z = np.linspace(-1, 1, 100)

# Create a meshgrid for evaluation
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Step 2: Evaluate the equation at each point
# Rearrange the equation to form: f(x, y, z) = 0, where f is the left hand side minus 1
F = (1 - np.sqrt(0.02 + X**2))**2 + Y**2 + Z**2 - 1

# Use a tolerance to find points close to the surface
tolerance = 0.001  # This can be adjusted based on how strict we want the condition to be

# Find indices where the condition is satisfied
indices = np.abs(F) < tolerance

# Extract the points that satisfy the equation
points = np.array([X[indices], Y[indices], Z[indices]])  # Transpose to get points in the correct shape

points # Show the shape of the points array and the first 10 points for inspection


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(*points)
plt.show()