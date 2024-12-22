import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class kmPoint:
    """Точка на плоскости"""

    def __init__(self, coord, id=""):
        """
        Конструктор точки.
        Вход:
            :coord: Координаты точки (список numpy из двух чисел).
            :id: Идентификатор точки (по умолчанию пустая строка).
        """
        self._coord = np.array(coord)
        self._id = id if id != "" else "Точка"

    @property
    def coord(self):
        """Возвращает координаты точки."""
        return self._coord
    
    @property
    def id(self):
        """Возвращает идентификатор точки."""
        return self._id

    @property
    def is_named(self):
        """Проверяет, имеет ли точка имя (идентификатор)."""
        return self._id != ""

    @classmethod
    def from_segment(cls, segment, t, id=""):
        """
        Конструктор точки, лежащей на прямой, содержащей отрезок.

        :segment: Экземпляр класса kmSegment.
        :t: Параметр точки на отрезке (в диапазоне [0, 1]).
        :id: Идентификатор точки (по умолчанию пустая строка).
        :return: Экземпляр класса kmPoint.
        """
        if not 0 <= t <= 1:
            raise ValueError("Параметр t должен быть в диапазоне [0, 1].")

        # Вычисление координат точки P(t) на отрезке
        coord = segment.parametric_equation(t)

        # Создаем объект точки с вычисленными координатами
        return cls(coord, id=id if id != "" else f"Point_on_{segment.id}")
    
    def __repr__(self):
        """Графическое представление точки"""
        x, y = self._coord
        plt.figure()
        plt.grid(True)
        plt.plot(x, y, "ro", markersize=10)
        plt.text(x + 0.03, y+0.01, self._id, fontsize=12, color='red')
        plt.xlim(x - 0.5, x + 0.5)
        plt.ylim(y - 0.5, y + 0.5)
        plt.title(f"Point: {self._id} at {tuple(self._coord)}")
        plt.show()

        return f"kmPoint(id={self._id}, coord={tuple(self._coord)})"


class kmVector:
    """Вектор на плоскости"""

    def __init__(self, coord, id=""):
        """
        Конструктор вектора.
        
        :coord: Координаты вектора (список ndarray).
        :id: Идентификатор вектора (по умолчанию пустая строка).
        """
        self._coord = np.array(coord)
        self._id = id if id != "" else "Вектор"

    @property
    def coord(self):
        """Возвращает координаты вектора."""
        return self._coord

    @property
    def id(self):
        """Возвращает идентификатор вектора."""
        return self._id

    @property
    def is_named(self):
        """Проверяет, имеет ли вектор ненулевой идентификатор."""
        return self._id != ""

    def __repr__(self):
        
        plt.figure()
        origin = np.array([0, 0])
        x, y = self._coord
        x_min = min(0, x) - 1
        x_max = max(0, x) + 1
        y_min = min(0, y) - 1
        y_max = max(0, y) + 1

        plt.quiver(*origin, x, y, angles='xy', scale_units='xy', scale=1, color='r', width=0.005)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.text(x * 0.5, y * 0.5, self._id, fontsize=12, color='blue')
        plt.grid(True)
        plt.title(f"Vector: {self._id} with coordinates {tuple(self._coord)}")
        plt.show()

        return f"kmVector(id={self._id}, coord={tuple(self._coord)})"

    @classmethod
    def from_points(cls, P0, P1, id=""):
        """
        Конструктор вектора по двум точкам.
        
        :P0: Начальная точка вектора (экземпляр kmPoint).
        :P1: Конечная точка вектора (экземпляр kmPoint).
        :id: Идентификатор вектора (по умолчанию пустая строка).
        """
        coord = np.array(P1.coord) - np.array(P0.coord)
        if id == "" and P0.is_named and P1.is_named:
            id = P0.id + P1.id
        return cls(coord, id)


class kmLine:
    """Прямая на плоскости"""

    def __init__(self, coef, id=""):
        """
        Конструктор прямой по коэффициентам уравнения.
        
        :param coef: Коэффициенты уравнения прямой (список или кортеж [A, B, C]).
        :param id: Идентификатор прямой (по умолчанию пустая строка).
        """
        self._coef = np.array(coef)
        self._id = id if id != "" else "Прямая"

    @property
    def coef(self):
        """Возвращает коэффициенты прямой (A, B, C)."""
        return self._coef

    @property
    def id(self):
        """Возвращает идентификатор прямой."""
        return self._id

    def equation(self):
        """Возвращает строковое представление уравнения прямой."""
        A, B, C = self._coef
        return f"{A}x + {B}y + {C} = 0"

    # Метод для получения направляющего вектора ("dir")
    @property
    def direction_vector(self):
        """Возвращает направляющий вектор прямой (идентификатор 'dir')."""
        # Направляющий вектор можно получить как [-B, A]
        A, B, _ = self._coef
        dir_vector = np.array([B, -A])
        return kmVector(dir_vector, id="dir")

    # Метод для получения нормального вектора ("norm")
    @property
    def normal_vector(self):
        """Возвращает нормальный вектор прямой (идентификатор 'norm')."""
        # Нормальный вектор — это вектор [A, B] из уравнения прямой
        A, B, _ = self._coef
        norm_vector = np.array([A, B])
        return kmVector(norm_vector, id="norm")
    
    def __repr__(self):
        """Графическое и текстовое представление прямой с помощью matplotlib."""
        plt.figure()

        # Настройка диапазона для осей
        x_vals = np.linspace(-10, 10, 400)
        A, B, C = self._coef

        if B != 0:
            y_vals = -(A * x_vals + C) / B
        else:
            x_vals = -C / A * np.ones_like(x_vals)
            y_vals = np.linspace(-10, 10, 400)

        plt.plot(x_vals, y_vals, label=f"{self.equation()}", color='b')

        # Настройка осей и сетки
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.grid(True)

        plt.title(f"Line: {self._id}")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()

        return f"kmLine(id={self._id}, coef={tuple(self._coef)})"

    @classmethod
    def from_points(cls, P1, P2, id=""):
        """
        Конструктор прямой по двум точкам.
        
        :param P1: Первая точка (экземпляр kmPoint).
        :param P2: Вторая точка (экземпляр kmPoint).
        :param id: Идентификатор прямой (по умолчанию пустая строка).
        """
        x1, y1 = P1.coord
        x2, y2 = P2.coord

        # Вычисляем коэффициенты прямой (Ax + By + C = 0)
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1

        if id == "" and P1.is_named and P2.is_named:
            id = P1.id + P2.id

        return cls([A, B, C], id)

    @classmethod
    def from_point_and_vector(cls, P, Dir, id=""):
        """
        Конструктор прямой по точке и направляющему вектору.
        
        :P: Точка на плоскости (экземпляр kmPoint).
        :Dir: Вектор направления (экземпляр kmVector).
        :id: Идентификатор прямой (по умолчанию пустая строка).
        """
        # Вектор нормали к направляющему вектору
        normal = np.array([-Dir.coord[1], Dir.coord[0]])  # [-y, x]
        A, B = normal
        C = -(A * P.coord[0] + B * P.coord[1])  # - (A*x0 + B*y0)

        if id == "" and P.is_named and Dir.is_named:
            id = P.id + Dir.id

        return cls([A, B, C], id)

    @classmethod
    def from_segment(cls, segment, id=""):
        """
        Конструктор прямой по отрезку.

        :segment: Экземпляр класса kmSegment, представляющий отрезок.
        :id: Идентификатор прямой (по умолчанию пустая строка).
        :return: Экземпляр класса kmLine.
        """
        # Получаем координаты точек A и B отрезка
        A = segment.P1.coord
        B = segment.P2.coord

        # Направляющий вектор от A до B
        direction_vector = B - A

        # Коэффициенты уравнения прямой: Ax + By + C = 0
        # Вектор нормали к прямой: (-dy, dx)
        A_coef = direction_vector[1]  # B (dy)
        B_coef = -direction_vector[0]  # -A (dx)

        # Подставляем точку A в уравнение прямой, чтобы найти C
        C_coef = -(A_coef * A[0] + B_coef * A[1])

        # Возвращаем объект прямой
        return cls([A_coef, B_coef, C_coef], id if id != "" else f"Line_on_{segment.id}")



class kmSegment:
    """Отрезок на плоскости, заданный двумя точками."""

    def __init__(self, P1, P2, id=""):
        """
        Конструктор отрезка по двум точкам.
        
        :P1: Первая точка (экземпляр kmPoint).
        :P2: Вторая точка (экземпляр kmPoint).
        :id: Идентификатор отрезка (по умолчанию пустая строка).
        """
        if np.array_equal(P1.coord, P2.coord):
            raise ValueError("Точки A и B должны быть различными для создания отрезка.")
        
        self.P1 = P1  # Точка A
        self.P2 = P2  # Точка B
        self._id = id if id != "" else "Segment"

    # Свойство "id" - возвращает идентификатор отрезка
    @property
    def id(self):
        return self._id

    # Вычисление длины отрезка
    @property
    def length(self):
        """Вычисляет длину отрезка."""
        return np.linalg.norm(self.P2.coord - self.P1.coord)

    # Вычисление средней точки отрезка
    @property
    def midpoint(self):
        """Возвращает координаты средней точки отрезка."""
        return kmPoint((self.P1.coord + self.P2.coord) / 2, id="midpoint")

    # Параметрическое уравнение отрезка
    def parametric_equation(self, t):
        """
        Возвращает координаты точки на отрезке в зависимости от параметра t.
        
        :t: Параметр, который должен быть в диапазоне [0, 1].
        :return: Координаты точки s(t).
        """
        if t < 0 or t > 1:
            raise ValueError("Параметр t должен быть в диапазоне [0, 1].")
        return self.P1.coord + (self.P2.coord - self.P1.coord) * t

    def __repr__(self):
        """Графическое и текстовое представление отрезка с помощью matplotlib."""
        plt.figure()

        # Визуализация отрезка
        x_vals = [self.P1.coord[0], self.P2.coord[0]]
        y_vals = [self.P1.coord[1], self.P2.coord[1]]
        plt.plot(x_vals, y_vals, 'ro-', label=f"{self.P1.id} to {self.P2.id}")

        # Визуализация средней точки
        midpoint = self.midpoint
        plt.plot(midpoint.coord[0], midpoint.coord[1], 'bo', label='Midpoint')

        # Настройка осей и сетки
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)

        # Добавление легенды
        plt.legend()

        # Показать график
        plt.title(f"Segment: {self._id} (Length: {self.length:.2f})")
        plt.xlim(min(x_vals) - 1, max(x_vals) + 1)
        plt.ylim(min(y_vals) - 1, max(y_vals) + 1)
        plt.show()

        # Возвращаем текстовое представление
        return f"kmSegment(id={self._id}, P1={tuple(self.P1.coord)}, P2={tuple(self.P2.coord)}, length={self.length:.2f})"

    @classmethod
    def from_parametric(cls, A, B, t1, t2, id=""):
        """
        Конструктор отрезка на основе параметрического уравнения прямой и значений параметров на концах.

        :A: Координаты начальной точки A.
        :B: Координаты конечной точки B.
        :t1: Значение параметра t для первой точки.
        :t2: Значение параметра t для второй точки.
        :id: Идентификатор отрезка (по умолчанию пустая строка).
        :return: Экземпляр класса kmSegment.
        """
        def parametric_func(t):
            """
            Внутренняя функция, задающая параметрическое уравнение прямой.
            :t: Параметр t в диапазоне [t1, t2].
            :return: Координаты точки на прямой.
            """
            return A.coord + (B.coord - A.coord) * t

        # Вычисляем координаты точек по параметрам t1 и t2
        P1_coords = parametric_func(t1)
        P2_coords = parametric_func(t2)

        # Создаем точки на основе этих координат
        P1 = kmPoint(P1_coords, id=f"Point_at_t_{t1}")
        P2 = kmPoint(P2_coords, id=f"Point_at_t_{t2}")

        # Возвращаем объект отрезка
        return cls(P1, P2, id)


