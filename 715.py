import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Параметры
r0 = 10  # Радиус неподвижной окружности
r = 3    # Радиус катящейся окружности
t = np.linspace(0, 2 * np.pi, 1000)  # Параметр для построения кривой

# Координаты неподвижной окружности
x_fixed = r0 * np.cos(t)
y_fixed = r0 * np.sin(t)

# Функция для вычисления координат эпициклоиды
def epicycloid(t, r0, r):
    x = (r0 + r) * np.cos(t) - r * np.cos((r0 + r) / r * t)
    y = (r0 + r) * np.sin(t) - r * np.sin((r0 + r) / r * t)
    return x, y

# Координаты эпициклоиды
x_epi, y_epi = epicycloid(t, r0, r)

# Создание фигуры и осей
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-(r0 + 2 * r), (r0 + 2 * r))
ax.set_ylim(-(r0 + 2 * r), (r0 + 2 * r))

# Начальные точки для визуализации
line_fixed, = ax.plot([], [], 'b-', label='Неподвижная окружность')
line_rolling, = ax.plot([], [], 'r-', label='Катящаяся окружность')
point, = ax.plot([], [], 'go', label='Фиксированная точка')
trace, = ax.plot([], [], 'g--', label='Траектория')

# Функция инициализации
def init():
    line_fixed.set_data(x_fixed, y_fixed)
    line_rolling.set_data([], [])
    point.set_data([], [])
    trace.set_data([], [])
    return line_fixed, line_rolling, point, trace

# Функция анимации
def animate(i):
    theta = t[i]
    x_center = (r0 + r) * np.cos(theta)
    y_center = (r0 + r) * np.sin(theta)
    
    # Координаты катящейся окружности
    x_rolling = x_center + r * np.cos(np.linspace(0, 2 * np.pi, 100))
    y_rolling = y_center + r * np.sin(np.linspace(0, 2 * np.pi, 100))
    
    # Координаты фиксированной точки
    x_point = x_center - r * np.cos((r0 + r) / r * theta)
    y_point = y_center - r * np.sin((r0 + r) / r * theta)
    
    # Обновление данных
    line_rolling.set_data(x_rolling, y_rolling)
    point.set_data(x_point, y_point)
    trace.set_data(x_epi[:i], y_epi[:i])
    
    return line_fixed, line_rolling, point, trace

# Создание анимации
ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init, interval=10, blit=True)

# Отображение легенды
ax.legend()

# Отображение графика
plt.show()