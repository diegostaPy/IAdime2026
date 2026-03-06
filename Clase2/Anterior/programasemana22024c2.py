import numpy as np
import matplotlib.pyplot as plt
N=100
x=2*np.random.rand(N, 1)
print(np.max(x))
print(np.min(x))
print(type(x))
print(np.shape(x))
y=3+2*x+(0.5*np.random.rand(N, 1)-0.25)

xl=np.linspace(0,2,10)
yl=3+2*xl
unos=np.ones((N,1))
A=np.hstack((unos,x))
print(A)
b=y
print(np.shape(A))
AtA=np.matmul(np.transpose(A),A)
invAtA=np.linalg.inv(AtA)
Atb=np.matmul(np.transpose(A),b)
w=np.matmul(invAtA,Atb)
print(w)

xe=xl
ye=w[0]+w[1]*xe

plt.plot(xl,yl)
plt.plot(xe,ye)
plt.scatter(x,y,alpha=0.3)
plt.show()




#y
