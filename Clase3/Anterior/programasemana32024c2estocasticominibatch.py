import numpy as np
import matplotlib.pyplot as plt

N=100
x=2*np.random.rand(N, 1)
y=3+2*x+(0.5*np.random.rand(N, 1)-0.25)

unos=np.ones((N,1))
A_or=np.hstack((unos,x))
b_or=y
    
lr=0.01
w=np.zeros((2,1))
mse=[]
minibatch=10
for i in range(100):
    for j in range(0, 100, minibatch):
        unos=np.ones((minibatch,1))
        A=np.hstack((unos,x[j:(j+minibatch)]))
        b = y[j:j+minibatch]
        Aw=np.matmul(A,w)
        Aw_bt=np.transpose(Aw-b)
        grad=2*np.matmul(Aw_bt,A)/minibatch
        w=w-lr*np.transpose(grad)
        print(i,w)
        ye=w[0]+w[1]*x
        error=(ye-y)**2
        mse.append(float(error.sum()))
plt.plot(mse)
plt.show()