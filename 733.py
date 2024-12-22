import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры
a = 2  # полуоси образующего эллипса
b = 1
r = 1  # радиус направляющей окружности

# Интервалы параметров
t = np.linspace(0, 3 * np.pi / 2, 100)
tau = np.linspace(0, np.pi, 100)
t, tau = np.meshgrid(t, tau)

# Уравнение поверхности эллиптического тора
x = (a * np.sin(tau) + r) * np.cos(t)
y = b * np.cos(tau)
z = -(a * np.sin(tau) + r) * np.sin(t)

# Визуализация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()