import numpy as np
n_puntos=int(input())
n_iteraciones=int(input())
lr =float(input())
# Generar datos de ejemplo
x = 10 * np.random.rand(n_puntos, 1) - 5
s = 5 * x + 0.1*np.random.randn(n_puntos, 1) + 3
p = 1 / (1 + np.exp(-s))
y=p> 0.5


A = np.c_[np.ones((len(x), 1)), x]

m=n_puntos


# Descenso del gradiente
w = np.random.randn(2,1)

for iteraciones in range(n_iteraciones):
    p=1/(1+np.exp(-A.dot( w)))
    if (iteraciones % 200 == 0):
        costo= (-1/m)*(np.sum((y*np.log(p)) + ((1-y)*(np.log(1-p)))))
        print(costo )
    gradiente =1/m * A.T.dot(p - y)
    w= w - lr* gradiente

print("Parámetros finales:", w)