import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load("../data/cylinder.npy")
data = data.reshape(151, 384, 199)

# Select the first time step
field_at_time_0 = data[5]  # Shape (384, 199)

# Plot the contour plot (without filling)
plt.figure(figsize=(10, 6))
contour = plt.contour(field_at_time_0, levels=20, colors='black')  # Create contour plot with 20 levels

# Add labels and title
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")


# Display the plot
plt.show()

