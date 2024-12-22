import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import random



class Poliedr:
    def __init__(self, data, faces):
        self.points = np.array(data)
        self.faces = np.array(faces)

    def get_face_vertices(self, face_index):
        face = self.faces[face_index]
        vertices = [self.points[i] for i in face]
        return vertices

    def get_face_normal(self, face_index):
        face = self.faces[face_index]
        q1, q2, q3 = self.points[face[0]], self.points[face[1]], self.points[face[2]]
        V = q2 - q1
        W = q3 - q2
        normal = np.cross(V, W)
        # Нормализуем вектор нормали
        normal = normal / np.linalg.norm(normal)
        return normal

    def is_convex(self):
        for face_index in range(len(self.faces)):
            face_normal = self.get_face_normal(face_index)
            face_vertices = self.get_face_vertices(face_index)
            q1, q2, q3 = face_vertices[0], face_vertices[1], face_vertices[2]
            
            # Вычисляем уравнение плоскости для текущей грани
            A, B, C = face_normal
            D = -np.dot(face_normal, q1)
            
            left_count = 0
            right_count = 0
            
            for pj in self.points:
                if np.array_equal(pj, q1) or np.array_equal(pj, q2) or np.array_equal(pj, q3):
                    continue  # Игнорируем вершины, лежащие на грани
                
                # Вычисляем положение точки pj относительно плоскости грани
                distance = A * pj[0] + B * pj[1] + C * pj[2] + D
                
                if distance > 0:
                    right_count += 1
                elif distance < 0:
                    left_count += 1
                
                # Если найдена пара вершин, лежащих по разные стороны от грани, полиэдр невыпуклый
                if left_count > 0 and right_count > 0:
                    return False
        
        # Если после проверки всех граней не найдено пар вершин, лежащих по разные стороны от грани, полиэдр выпуклый
        return True

def generate_dodecahedron():
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение
    vertices = np.array([
        (+1, +1, +1), (+1, +1, -1), (+1, -1, +1), (+1, -1, -1),
        (-1, +1, +1), (-1, +1, -1), (-1, -1, +1), (-1, -1, -1),
        (0, +phi, +1/phi), (0, +phi, -1/phi), (0, -phi, +1/phi), (0, -phi, -1/phi),
        (+1/phi, 0, +phi), (+1/phi, 0, -phi), (-1/phi, 0, +phi), (-1/phi, 0, -phi),
        (+phi, +1/phi, 0), (+phi, -1/phi, 0), (-phi, +1/phi, 0), (-phi, -1/phi, 0)
    ])
    faces = [
        [0, 12, 2, 17, 16], [0, 16, 1, 9, 8], [0, 12, 14, 4, 8], [12, 14, 6, 10, 2],
        [2, 10, 11, 3, 17], [17, 16, 1, 13, 3], [1, 9, 5, 15, 13], [3, 11, 7, 15, 13],
        [8, 4, 18, 5, 9], [5, 18, 19, 7, 15], [7, 19, 6, 10, 11], [6, 14, 4, 18, 19]
    ]

    return vertices, faces
vertices, faces = generate_dodecahedron()
def calculate_edges(faces):
    edges = set()
    for face in faces:
        for i in range(len(face)):
            edge = tuple(sorted((face[i], face[(i + 1) % len(face)])))  # Пары вершин
            edges.add(edge)
    return list(edges)

edges = calculate_edges(faces)

# Создание 3D-графика
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])  # Пропорции графика

edge_lines = [[vertices[edge[0]], vertices[edge[1]]] for edge in edges]
ax.add_collection3d(Line3DCollection(edge_lines, color='k', linewidths=1))

for face in faces:
    polygon = vertices[face]
    poly3d = [[list(p) for p in polygon]]
    color = plt.cm.tab20(random.random())  # Случайный цвет
    ax.add_collection3d(Poly3DCollection(poly3d, facecolor=color, edgecolor='k', alpha=0.5, linewidths=1))

ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=20)
for idx, (x, y, z) in enumerate(vertices):
    ax.text(x, y, z, str(idx), color='blue', fontsize=30, ha='center', va='center')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("додекаэдр")
plt.show()
