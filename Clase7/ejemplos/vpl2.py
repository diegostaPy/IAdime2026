import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Número de puntos y número de iteraciones
n_puntos = int(input())
n_iteraciones = int(input())


# Generar datos de ejemplo
x = 10 * np.random.rand(n_puntos, 1) - 5
s = 5 * x + 0.1 * np.random.randn(n_puntos, 1) + 3
p = 1 / (1 + np.exp(-s))
y = p > 0.5

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Inicializar y ajustar el modelo de regresión logística
modelo = LogisticRegression(fit_intercept=True,max_iter=n_iteraciones)
modelo.fit(X_train, y_train)
print(modelo.intercept_)

print(modelo.coef_)
# Evaluar el rendimiento del modelo
precision = modelo.score(X_test, y_test)
print("Precisión del modelo:", precision)