import numpy as np
import time
import random
from collections import defaultdict
from math import sqrt


class Node:

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:

    def __init__(self):
        self.root = None

    def insert(self, key, value):
        self.root = self._insert_recursive(self.root, key, value)

    def _insert_recursive(self, node, key, value):
        if not node:
            return Node(key, value)
        elif key < node.key:
            node.left = self._insert_recursive(node.left, key, value)
        else:
            node.right = self._insert_recursive(node.right, key, value)

        node.height = 1 + max(self._get_height(node.left),
                              self._get_height(node.right))
        balance = self._get_balance(node)

        if balance > 1 and key < node.left.key:
            return self._rotate_right(node)
        if balance < -1 and key > node.right.key:
            return self._rotate_left(node)
        if balance > 1 and key > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1 and key < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def search(self, key):
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node, key):
        if node is None or node.key == key:
            return node
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)


class RatingReader:

    def __init__(self, filename):
        self.filename = filename

    def read_ratings(self):
        ratings = defaultdict(dict)
        with open(self.filename, 'r', encoding='utf-8') as file:
            next(file)
            for line in file:
                parts = line.strip().split(',')
                user_id, movie_id, rating, _ = parts
                ratings[int(user_id)][int(movie_id)] = float(rating)
        return ratings


class MovieReader:

    def __init__(self, filename):
        self.filename = filename

    def read_movies(self):
        movies = {}
        with open(self.filename, 'r', encoding='utf-8') as file:
            next(file)
            for line in file:
                parts = line.strip().split(',')
                movie_id, title, genres = parts[0], parts[1], parts[2]
                movies[int(movie_id)] = {'title': title, 'genres': genres}
        return movies


def calcular_distancia(personaA, personaB, tipo, tree):
    node_personaA = tree.search(personaA)
    node_personaB = tree.search(personaB)

    if node_personaA is None or node_personaB is None:
        return None

    valores_personaA = node_personaA.value
    valores_personaB = node_personaB.value
    indices_comunes = np.intersect1d(np.where(~np.isnan(valores_personaA)),
                                     np.where(~np.isnan(valores_personaB)))
    valores_personaA = valores_personaA[indices_comunes]
    valores_personaB = valores_personaB[indices_comunes]

    if tipo == "manhattan":
        return np.sum(np.abs(valores_personaA - valores_personaB))
    elif tipo == "euclidiana":
        return sqrt(np.sum((valores_personaA - valores_personaB)**2))
    elif tipo == "pearson":
        if len(valores_personaA) < 2:
            return np.nan

        mean_personaA = np.mean(valores_personaA)
        mean_personaB = np.mean(valores_personaB)

        numerator = np.sum((valores_personaA - mean_personaA) * (valores_personaB - mean_personaB))
        denominator = sqrt(np.sum((valores_personaA - mean_personaA)**2) * np.sum((valores_personaB - mean_personaB)**2))

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    elif tipo == "coseno":
        dot_product = np.dot(valores_personaA, valores_personaB)
        norm_A = np.linalg.norm(valores_personaA)
        norm_B = np.linalg.norm(valores_personaB)
        if norm_A != 0 and norm_B != 0:
            return dot_product / (norm_A * norm_B)
        else:
            return 0


def knn(usuario, distancia, k, tree, ratings, movies):
    distancias = {}
    stack = []
    current = tree.root

    while current or stack:
        while current:
            if current.key != usuario:
                dist_actual = calcular_distancia(usuario, current.key,
                                                 distancia, tree)
                if dist_actual is not None:
                    distancias[current.key] = dist_actual
            stack.append(current)
            current = current.left

        current = stack.pop()
        current = current.right

    vecinos_cercanos = sorted(distancias.items(), key=lambda x: x[1])[:k]

    if not vecinos_cercanos:
        print("No se encontraron vecinos cercanos.")
        return

    print("\n***** Resultados de Vecinos más Cercanos *****\n")
    print("a) Todos los vecinos más cercanos:")
    for vecino, dist in vecinos_cercanos:
        dist = abs(dist)
        print(f"Usuario {vecino} con una distancia de {dist:.4f}")

    coincidencias_exactas = defaultdict(int)
    coincidencias_positivas = defaultdict(int)
    coincidencias_negativas = defaultdict(int)

    for vecino, _ in vecinos_cercanos:
        for movie_id, rating in ratings[vecino].items():
            if movie_id in ratings[usuario]:
                if ratings[usuario][movie_id] == rating:
                    coincidencias_exactas[movie_id] += 1
                if ratings[usuario][movie_id] >= 3 and rating >= 3:
                    coincidencias_positivas[movie_id] += 1
                if ratings[usuario][movie_id] <= 3 and rating <= 3:
                    coincidencias_negativas[movie_id] += 1

    print("\nb) Número de coincidencias exactas:")
    if coincidencias_exactas:
        for movie_id, count in coincidencias_exactas.items():
            title = movies.get(movie_id, {}).get('title', 'Desconocido')
            genres = movies.get(movie_id, {}).get('genres', 'Desconocido')
            print(
                f"Película: {title}, Categoría: {genres}, Coincidencias exactas: {count}"
            )
    else:
        print("No hay coincidencias exactas.")

    print("\nc) Número de coincidencias positivas(+):")
    if coincidencias_positivas:
        for movie_id, count in coincidencias_positivas.items():
            title = movies.get(movie_id, {}).get('title', 'Desconocido')
            genres = movies.get(movie_id, {}).get('genres', 'Desconocido')
            print(
                f"Película: {title}, Categoría: {genres}, Coincidencias positivas: {count}"
            )
    else:
        print("No hay coincidencias positivas.")

    print("\nd) Número de coincidencias negativas(-):")
    if coincidencias_negativas:
        for movie_id, count in coincidencias_negativas.items():
            title = movies.get(movie_id, {}).get('title', 'Desconocido')
            genres = movies.get(movie_id, {}).get('genres', 'Desconocido')
            print(
                f"Película: {title}, Categoría: {genres}, Coincidencias negativas: {count}"
            )
    else:
        print("No hay coincidencias negativas.")

    print("\nD) El ranking de los 3 más similares al usuario:")
    usuario_ranking = distancias[vecinos_cercanos[0][0]]
    print(
        f"Usuario más similar: {vecinos_cercanos[0][0]} con una distancia de {abs(usuario_ranking):.4f}"
    )
    for i, (vecino, dist) in enumerate(vecinos_cercanos[1:4], start=2):
        dist = abs(dist)
        print(
            f"{i}) Usuario {vecino} con una distancia de {dist:.4f}"
        )

    print("\nE) Recomendar las películas:")
    mean_distance = sum(dist for _, dist in vecinos_cercanos) / k
    for movie_id, rating in ratings[usuario].items():
        if movie_id not in ratings[vecinos_cercanos[0][0]]:
            recommendation_distance = mean_distance + random.uniform(-0.1, 0.1)
            recommendation_distance = abs(recommendation_distance)
            title = movies.get(movie_id, {}).get('title', 'Desconocido')
            genres = movies.get(movie_id, {}).get('genres', 'Desconocido')
            print(
                f"Película: {title}, Categoría: {genres}, Rating: {rating}, distancia: {recommendation_distance:.4f} (Usuario seleccionado)"
            )

def knn_calculated(usuario_id, k, tipo_distancia, tree, ratings, movies):
    distancias = {}
    stack = []
    current = tree.root

    while current or stack:
        while current:
            if current.key != usuario_id:
                dist_actual = calcular_distancia(usuario_id, current.key, tipo_distancia, tree)
                if dist_actual is not None:
                    distancias[current.key] = dist_actual
            stack.append(current)
            current = current.left

        current = stack.pop()
        current = current.right

    vecinos_cercanos = sorted(distancias.items(), key=lambda x: x[1])[:k]

    if not vecinos_cercanos:
        print("No se encontraron vecinos cercanos.")
        return

    print("\n***** Vecinos más cercanos *****\n")
    for vecino, dist in vecinos_cercanos:
        dist = abs(dist)
        print(f"Usuario {vecino} con una distancia de {dist:.4f}")


def main():
    print("\n*** Sistema de recomendaciones ***\n")

    rating_reader = RatingReader('C:/Users/DELL/Desktop/Bases de datos completa/ml-20m/ratings.csv')
    ratings = rating_reader.read_ratings()

    movie_reader = MovieReader('C:/Users/DELL/Desktop/Bases de datos completa/ml-20m/movies.csv')
    movies = movie_reader.read_movies()

    tree = AVLTree()
    for user_id, ratings_user in ratings.items():
        tree.insert(user_id, np.array(list(ratings_user.values())))

    while True:
        print("\n***** Menú de opciones *****\n")
        print("1. Calcular la distancia de Manhattan")
        print("2. Calcular la distancia Euclidiana")
        print("3. Calcular aproximación de Pearson")
        print("4. Calcular la similitud del coseno")
        print("5. Encontrar vecinos más cercanos")
        print("6. Salir")
        opcion = input("\nTipo de opción: ")
        print("\n+-------------------------------------------+\n")

        if opcion in ["1", "2", "3", "4"]:
            tipo_distancia = "manhattan" if opcion == "1" else "euclidiana" if opcion == "2" else "pearson" if opcion == "3" else "coseno"
            print("\nIngresar IDs de las personas que desea:\n")
            usuarioA = int(input("- Persona A: "))
            usuarioB = int(input("- Persona B: "))

            start_time = time.time()
            distancia = calcular_distancia(usuarioA,
                                           usuarioB,
                                           tipo=tipo_distancia,
                                           tree=tree)
            execution_time = time.time() - start_time

            print(
                f"\n- La {('similitud' if tipo_distancia in ['pearson', 'coseno'] else 'distancia')} {tipo_distancia.title()} entre {usuarioA} y {usuarioB} es de: {distancia:.4f}"
            )
            print(f"Tiempo de ejecución: {execution_time:.4f} segundos.")

        elif opcion == "5":
            while True:
                print(
                    "\n***** Menú de opciones de Vecinos más Cercanos *****\n")
                print("a) Calcula KNN con un solo usuario")
                print("b) Calcular con dos usuarios nuevos")
                print("c) Recomendar peliculas")
                print("d) Regresar al menú principal")
                opcion_vecinos = input("\nSeleccione una opción (a, b, c o d): ")

                if opcion_vecinos == "a":
                    usuario_id = int(input("\nID de Usuario: "))
                    k = int(input("Número de vecinos (k): "))
                    tipo_distancia = input("Tipo de distancia (manhattan, euclidiana, pearson, coseno): ")

                    start_time = time.time()
                    knn_calculated(usuario_id, k, tipo_distancia, tree, ratings, movies)
                    execution_time = time.time() - start_time

                    print(f"Tiempo de ejecución: {execution_time:.4f} segundos.")
                    break


                elif opcion_vecinos == "b":
                    print("\nIngresar información del primer usuario:\n")
                    while True:
                        usuarioA_id = int(
                            input("- ID del primer usuario (nuevo): "))
                        if usuarioA_id not in ratings:
                            break
                        else:
                            print(
                                "Este ID ya está en uso. Por favor, elija otro."
                            )

                    n_peliculas_A = int(
                        input("- Número de películas vistas: "))
                    peliculas_A = []
                    calificaciones_A = []
                    for i in range(n_peliculas_A):
                        while True:
                            movie_id = int(input(f"- ID de película {i+1}: "))
                            if movie_id in ratings[usuarioA_id]:
                                print(
                                    "Este ID ya está en uso. Por favor, elija otro."
                                )
                            else:
                                break
                        rating = float(
                            input(f"- Calificación de película {i+1}: "))
                        peliculas_A.append(movie_id)
                        calificaciones_A.append(rating)

                    print("\nIngresar información del segundo usuario:\n")
                    while True:
                        usuarioB_id = int(
                            input("- ID del segundo usuario (nuevo): "))
                        if usuarioB_id not in ratings:
                            break
                        else:
                            print(
                                "Este ID ya está en uso. Por favor, elija otro."
                            )

                    n_peliculas_B = int(
                        input("- Número de películas vistas: "))
                    peliculas_B = []
                    calificaciones_B = []
                    for i in range(n_peliculas_B):
                        while True:
                            movie_id = int(input(f"- ID de película {i+1}: "))
                            if movie_id in ratings[usuarioB_id]:
                                print(
                                    "Este ID ya está en uso. Por favor, elija otro."
                                )
                            else:
                                break
                        rating = float(
                            input(f"- Calificación de película {i+1}: "))
                        peliculas_B.append(movie_id)
                        calificaciones_B.append(rating)

                    # Añadir las calificaciones a la base de datos
                    for i in range(n_peliculas_A):
                        ratings[usuarioA_id][
                            peliculas_A[i]] = calificaciones_A[i]
                    for i in range(n_peliculas_B):
                        ratings[usuarioB_id][
                            peliculas_B[i]] = calificaciones_B[i]

                    # Insertar los nuevos usuarios al árbol
                    tree.insert(usuarioA_id, np.array(calificaciones_A))
                    tree.insert(usuarioB_id, np.array(calificaciones_B))

                    k = int(input("\nNúmero de vecinos (k): "))
                    tipo_distancia = input(
                        "Tipo de distancia (manhattan, euclidiana, pearson, coseno): "
                    )

                    start_time = time.time()
                    knn(usuarioA_id, tipo_distancia, k, tree, ratings, movies)
                    execution_time = time.time() - start_time

                    print(
                        f"Tiempo de ejecución: {execution_time:.4f} segundos.")
                    break

                elif opcion_vecinos == "c":
                    usuario_id = int(input("\nID de Usuario: "))
                    k = 10
                    tipo_distancia = "coseno"

                    start_time = time.time()
                    knn(usuario_id, tipo_distancia, k, tree, ratings, movies)
                    execution_time = time.time() - start_time

                    print(
                        f"Tiempo de ejecución: {execution_time:.4f} segundos.")
                    break

                elif opcion_vecinos == "d":
                    break

                else:
                    print("Opción no válida. Por favor, seleccione a, b, c o d.")

        elif opcion == "6":
            print("\n¡Hasta luego!")
            break

        else:
            print(
                "Opción no válida. Por favor, seleccione una opción del menú.")


if __name__ == "__main__":
    main()

