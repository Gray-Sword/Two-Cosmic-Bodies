import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
m1 = 5.97e24     # mass of the first particle (kg) - approx mass of Earth
m2 = 7.35e22     # mass of the second particle (kg) - approx mass of Moon
dt = 100         # time step (s)

# Initial positions and velocities
pos1 = np.array([0, 0], dtype=float)      # position of the first particle (m)
pos2 = np.array([3.84e8, 0], dtype=float) # position of the second particle (m)

vel1 = np.array([0, 0], dtype=float)      # initial velocity of the first particle (m/s)
vel2 = np.array([0, 1023], dtype=float)   # initial velocity of the second particle (m/s)

# Lists to store positions for plotting
positions1 = [pos1.copy()]
positions2 = [pos2.copy()]

# Function to compute gravitational force
def gravitational_force(m1, m2, pos1, pos2):
    r = np.linalg.norm(pos2 - pos1)
    force_magnitude = G * m1 * m2 / r**2
    force_direction = (pos2 - pos1) / r
    return force_magnitude * force_direction

# Simulation loop
num_steps = 1000
for _ in range(num_steps):
    # Calculate gravitational force
    force = gravitational_force(m1, m2, pos1, pos2)

    # Update velocities based on the force
    vel1 += force / m1 * dt
    vel2 -= force / m2 * dt

    # Update positions based on velocities
    pos1 += vel1 * dt
    pos2 += vel2 * dt

    # Store positions for plotting
    positions1.append(pos1.copy())
    positions2.append(pos2.copy())

# Convert to numpy arrays for easy plotting
positions1 = np.array(positions1)
positions2 = np.array(positions2)

# Plotting the result
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(positions1[:, 0], positions1[:, 1], label="Particle 1 (m1)")
ax.plot(positions2[:, 0], positions2[:, 1], label="Particle 2 (m2)")
ax.legend()
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
plt.title("Two-Body Orbital Motion under Gravitational Force")
plt.show()
