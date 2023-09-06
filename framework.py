import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Cargar datos
df1 = load_digits()
df2 = load_iris()

# Dividir datos de entrada y de salida para digits
X_1 = df1.data
y_1 = df1.target

# Dividir datos de entrada y de salida para digits
X_2 = df2.data
y_2 = df2.target

# Dividir los datos en conjunto de entrenamiento (70%) y conjunto de prueba (30%) para digits con una semilla random
# Conjunto 1
X1_train, X1_test, y1_train, y1_test = train_test_split(X_1, y_1, test_size=0.3, random_state=random.randint(1,100))
# Conjunto 2
X3_train, X3_test, y3_train, y3_test = train_test_split(X_1, y_1, test_size=0.3, random_state=random.randint(1,100))

# Dividir los datos en conjunto de entrenamiento (70%) y conjunto de prueba (30%) para iris con una semilla random
# Conjunto 1
X2_train, X2_test, y2_train, y2_test = train_test_split(X_2, y_2, test_size=0.3, random_state=random.randint(1,100))
# Conjunto 2
X4_train, X4_test, y4_train, y4_test = train_test_split(X_2, y_2, test_size=0.3, random_state=random.randint(1,100))


# Pasar datos de salida esperados de las pruebas a lista
y1_test = y1_test.tolist()
y2_test = y2_test.tolist()
y3_test = y3_test.tolist()
y4_test = y4_test.tolist()

"""
Cargar modelo knn en variables para su futuro uso, se eligió el modelo knn ya que es bueno para problemas de clasificación
como el que proponen los datasets de digits e iris
"""
knn_digits1 = KNeighborsClassifier(n_neighbors=3)
knn_digits1.fit(X1_train, y1_train)
knn_iris1 = KNeighborsClassifier(n_neighbors=3)
knn_iris1.fit(X2_train, y2_train)
knn_digits2 = KNeighborsClassifier(n_neighbors=3)
knn_digits2.fit(X3_train, y3_train)
knn_iris2 = KNeighborsClassifier(n_neighbors=3)
knn_iris2.fit(X4_train, y4_train)


# Realizar predicciones en los datos de prueba

# Realiza predicciones para el dataset digits 1
predictions_digits = knn_digits1.predict(X1_test)
# Realiza predicciones para le dataset iris 1
predictions_iris = knn_iris1.predict(X2_test)
# Medir precisión del modelo con dataset digits 1
precision1 = accuracy_score(y1_test, predictions_digits)
# Medir precisión del modelo con dataset iris 1
precision2 = accuracy_score(y2_test, predictions_iris)
# Matriz de confusion del modelo con dataset digits 1
conf_matrix1 = confusion_matrix(y1_test, predictions_digits)
# Matriz de confusion del modelo con dataset iris 1
conf_matrix2 = confusion_matrix(y2_test, predictions_iris)

# Realiza predicciones para el dataset digits 2
predictions_digits2 = knn_digits2.predict(X3_test)
# Realiza predicciones para le dataset iris 2
predictions_iris2 = knn_iris2.predict(X4_test)
# Medir precisión del modelo con dataset digits 2
precision3 = accuracy_score(y3_test, predictions_digits2)
# Medir precisión del modelo con dataset iris 2
precision4 = accuracy_score(y4_test, predictions_iris2)
# Matriz de confusion del modelo con dataset digits 2
conf_matrix3 = confusion_matrix(y3_test, predictions_digits2)
# Matriz de confusion del modelo con dataset iris 2
conf_matrix4 = confusion_matrix(y4_test, predictions_iris2)

# Impresión de los datos y métricas en consola
print("Predicciones digits:", predictions_digits)
print("\nReales digits\n", y1_test)
print("\nPrecision digits\n", precision1)
print("\nMatriz de confusion digits: \n", conf_matrix1)
print(classification_report(y1_test, predictions_digits))
print("\nPredicciones iris:\n", predictions_iris)
print("\nReales iris\n", y2_test)
print("\nPrecision iris\n", precision2)
print("\nMatriz de confusion iris: \n", conf_matrix2)
print(classification_report(y2_test, predictions_iris))
print("\nPredicciones digits 2:\n", predictions_digits2)
print("\nReales digits 2\n", y3_test)
print("\nPrecision digits\n", precision3)
print("\nMatriz de confusion digits: \n", conf_matrix3)
print(classification_report(y3_test, predictions_digits2))
print("\nPredicciones iris:\n", predictions_iris2)
print("\nReales iris\n", y4_test)
print("\nPrecision iris\n", precision4)
print("\nMatriz de confusion iris: \n", conf_matrix4)
print(classification_report(y4_test, predictions_iris2))
