import matplotlib.pyplot as plt
import numpy as np
import csv
import random

# Función para leer el archivo CSV y separar datos de etiquetas
def cargar_datos(archivo_csv):
    datos = []
    etiquetas = []
    try:
        with open(archivo_csv, 'r') as archivo:
            lector_csv = csv.reader(archivo)
            next(lector_csv)  # Saltar la primera fila de encabezados
            for fila in lector_csv:
                try:
                    caracteristica1 = float(fila[0])
                    caracteristica2 = float(fila[1])
                    etiqueta = float(fila[2])  # Convertir la etiqueta a float
                    datos.append([caracteristica1, caracteristica2])
                    etiquetas.append(etiqueta)
                except (ValueError, IndexError) as e:
                    print(f"Error al procesar la fila {fila}: {e}")
    except FileNotFoundError:
        print(f"Archivo {archivo_csv} no encontrado.")
    return datos, etiquetas

# Función de entrenamiento del perceptrón
def entrenar_perceptron(datos, etiquetas, tasa_aprendizaje=0.1, epocas=100):
    num_caracteristicas = len(datos[0])
    pesos = [random.uniform(0, 1) for _ in range(num_caracteristicas)]
    umbral = random.uniform(0, 1)

    for _ in range(epocas):
        for i in range(len(datos)):
            prediccion = sum(pesos[j] * datos[i][j] for j in range(num_caracteristicas)) + umbral
            if prediccion >= 0:
                salida = 1
            else:
                salida = -1
            error = etiquetas[i] - salida
            for j in range(num_caracteristicas):
                pesos[j] += tasa_aprendizaje * error * datos[i][j]
            umbral += tasa_aprendizaje * error

    return pesos, umbral

# Función de predicción
def predecir(datos, pesos, umbral):
    prediccion = []
    for dato in datos:
        suma = sum(pesos[j] * dato[j] for j in range(len(dato))) + umbral
        if suma >= 0:
            prediccion.append(1)
        else:
            prediccion.append(-1)
    return prediccion

# Función para evaluar la precisión del perceptrón
def evaluar_precision(etiquetas_reales, etiquetas_predichas):
    correctos = sum(1 for r, p in zip(etiquetas_reales, etiquetas_predichas) if r == p)
    precision = correctos / len(etiquetas_reales)
    return precision

# Función para visualizar los datos en 2D
def visualizar_datos(datos, etiquetas):
    plt.scatter([dato[0] for dato, etiqueta in zip(datos, etiquetas) if etiqueta == 1],
                [dato[1] for dato, etiqueta in zip(datos, etiquetas) if etiqueta == 1],
                label='Categoría A', marker='o')

    plt.scatter([dato[0] for dato, etiqueta in zip(datos, etiquetas) if etiqueta == -1],
                [dato[1] for dato, etiqueta in zip(datos, etiquetas) if etiqueta == -1],
                label='Categoría B', marker='x')

    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Visualización de Datos en 2D')
    plt.legend()
    plt.show()

# Función para visualizar los datos en 2D junto con el hiperplano
def visualizar_hiperplano(datos, etiquetas, pesos, umbral):
    plt.scatter([dato[0] for dato, etiqueta in zip(datos, etiquetas) if etiqueta == 1],
                [dato[1] for dato, etiqueta in zip(datos, etiquetas) if etiqueta == 1],
                label='Categoría A', marker='o')

    plt.scatter([dato[0] for dato, etiqueta in zip(datos, etiquetas) if etiqueta == -1],
                [dato[1] for dato, etiqueta in zip(datos, etiquetas) if etiqueta == -1],
                label='Categoría B', marker='x')

    # Dibujar el hiperplano
    x_min, x_max = min([dato[0] for dato in datos]), max([dato[0] for dato in datos])
    y_min, y_max = min([dato[1] for dato in datos]), max([dato[1] for dato in datos])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    hiperplano = pesos[0] * xx + pesos[1] * yy + umbral
    plt.contour(xx, yy, hiperplano, levels=[0], colors='k', linestyles='dashed')

    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Visualización de Datos y Hiperplano en 2D')
    plt.legend()
    plt.show()

# Cargar los datos desde el archivo CSV
archivo_csv = 'C:/Users/Luz Karen/Desktop/perceptron_tasks.csv'
datos, etiquetas = cargar_datos(archivo_csv)

# Visualizar los datos en 2D antes de entrenar el perceptrón
visualizar_datos(datos, etiquetas)

# Dividir los datos en entrenamiento y pruebas (puedes ajustar la proporción)
datos_entrenamiento = datos[:800]
etiquetas_entrenamiento = etiquetas[:800]
datos_pruebas = datos[800:]
etiquetas_pruebas = etiquetas[800:]

# Entrenar el perceptrón
pesos, umbral = entrenar_perceptron(datos_entrenamiento, etiquetas_entrenamiento)

# Predecir con los datos de prueba
etiquetas_predichas = predecir(datos_pruebas, pesos, umbral)

# Evaluar la precisión
precision = evaluar_precision(etiquetas_pruebas, etiquetas_predichas)
print("Precisión del perceptrón:", precision)

# Visualizar los datos y el hiperplano aprendido
visualizar_hiperplano(datos, etiquetas, pesos, umbral)


