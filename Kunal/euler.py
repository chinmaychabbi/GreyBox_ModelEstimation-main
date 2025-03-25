import numpy as np
import matplotlib.pyplot as plt

# Constants
a1 = -0.1
b1 = 0.1
f1 = 10  # Hz
a2 = -0.2
b2 = 0.1
f2 = 20  # Hz
dt = 0.001  # Time step for Euler's method, 1ms
t_final = 1  # Final time in seconds
n_steps = int(t_final / dt)  # Number of steps

# Time vector
t = np.linspace(0, t_final, n_steps)

# Initialize state vectors
x1 = np.zeros(n_steps)
x2 = np.zeros(n_steps)

# Euler method for system of differential equations
for i in range(1, n_steps):
    u1 = 0.1 * np.sin(2 * np.pi * f1 * t[i-1])
    u2 = 0.1 * np.cos(2 * np.pi * f2 * t[i-1])
    
    x1[i] = x1[i-1] + dt * (a1 * x1[i-1] + b1 * u1 + x2[i-1])
    x2[i] = x2[i-1] + dt * (a2 * x2[i-1] + b2 * u2 + x1[i-1])

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, x1, label='$x_1$')
plt.title('Trajectories of $x_1$ and $x_2$')
plt.ylabel('$x_1$')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, x2, color='orange', label='$x_2$')
plt.xlabel('Time (s)')
plt.ylabel('$x_2$')
plt.grid(True)

plt.show()
