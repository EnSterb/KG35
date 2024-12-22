import matplotlib.pyplot as plt
import random

def check_point_in_circle(point, tri_vertex1, tri_vertex2, tri_vertex3):
    """Проверяет, находится ли точка внутри окружности, описанной вокруг треугольника."""
    x, y = point
    ax, ay = tri_vertex1
    bx, by = tri_vertex2
    cx, cy = tri_vertex3

    matrix = [
        [ax - x, ay - y, (ax - x) ** 2 + (ay - y) ** 2],
        [bx - x, by - y, (bx - x) ** 2 + (by - y) ** 2],
        [cx - x, cy - y, (cx - x) ** 2 + (cy - y) ** 2],
    ]

    det = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
        matrix[1][0] * (matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]) +
        matrix[2][0] * (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1])
    )
    return det > 0

def visualize(points, triangulation, final=False, pause=0.5):
    """Показывает триангуляцию в реальном времени."""
    plt.clf()
    for tri in triangulation:
        x_coords = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
        y_coords = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
        plt.plot(x_coords, y_coords, 'g-' if final else 'b-')
    px, py = zip(*points)
    plt.plot(px, py, 'ro')
    if final:
        plt.title("Результирующая триангуляция")
    plt.pause(pause)

def delaunay_triangulation(input_points):
    """Реализует алгоритм Делоне для множества точек."""
    # Создание супертреугольника
    min_x = min(p[0] for p in input_points)
    max_x = max(p[0] for p in input_points)
    min_y = min(p[1] for p in input_points)
    max_y = max(p[1] for p in input_points)

    delta = max(max_x - min_x, max_y - min_y)
    super_triangle = [
        (min_x - 2 * delta, min_y - delta),
        (max_x + 2 * delta, min_y - delta),
        (min_x + (max_x - min_x) // 2, max_y + 2 * delta)
    ]

    # Инициализация с супертреугольником
    triangulation = [tuple(super_triangle)]

    plt.ion()
    visualize(input_points, triangulation, pause=1.0)

    for current_point in input_points:
        invalid_tris = []

        # Вычисляем "плохие" треугольники
        for tri in triangulation:
            if check_point_in_circle(current_point, *tri):
                invalid_tris.append(tri)

        # Определяем границы образовавшегося многоугольника
        boundary_edges = []
        for tri in invalid_tris:
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                if edge[::-1] in boundary_edges:
                    boundary_edges.remove(edge[::-1])
                elif edge not in boundary_edges:
                    boundary_edges.append(edge)

        # Убираем плохие треугольники
        triangulation = [tri for tri in triangulation if tri not in invalid_tris]

        # Добавляем новые треугольники, образованные текущей точкой
        for edge in boundary_edges:
            triangulation.append((edge[0], edge[1], current_point))

        visualize(input_points, triangulation)

    # Убираем треугольники, связанные с вершинами супертреугольника
    vertices_to_remove = set(super_triangle)
    final_triangulation = [
        tri for tri in triangulation
        if not any(vertex in vertices_to_remove for vertex in tri)
    ]

    visualize(input_points, final_triangulation, final=True, pause=1.0)
    plt.ioff()
    plt.show()

    return final_triangulation

# Тестовый пример
if __name__ == "__main__":
    # Генерация случайных точек
    random.seed(42)
    num_points = 10
    test_points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)]

    output = delaunay_triangulation(test_points)
    print("Результирующие треугольники:")
    for tri in output:
        print(tri)
