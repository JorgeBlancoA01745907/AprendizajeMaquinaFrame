Autor: Jorge Isidro Blanco Martínez
# k-NN para Clasificación de Dígitos Escritos a Mano

Este repositorio contiene un ejemplo de implementación del algoritmo k-NN (k-Nearest Neighbors) en Python para la clasificación de dígitos utilizando la libreria neighbors de sklearn

## Contenido

- `framework.py`: El archivo principal que contiene el código para cargar los datos, utilizar el modelo knn de sklearn, realizar predicciones y mostrar los resultados.

## Requisitos

1. Clonar repositorio de Github
2. Asegúrate de tener Python instalado en tu sistema.
3. Instala las bibliotecas necesarias usando el siguiente comando:

```
pip install scikit-learn
```

## Uso

1. Ejecuta el archivo `framework.py`.

El script realizará las siguientes acciones:

1. Cargará el conjunto de datos de dígitos escritos a mano utilizando `load_digits()` de scikit-learn.
2. Dividirá los datos en conjuntos de entrenamiento (70%) y prueba (30%).
3. Cargara el modelo knn en variables
4. Realizará predicciones en los datos de prueba utilizando el algoritmo k-NN.
5. Mostrará las predicciones generadas por el algoritmo y las etiquetas reales.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---