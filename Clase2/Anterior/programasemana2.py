import matplotlib.pyplot as plt
import numpy as np

arr = np.array([[1, 2], [3, 4]])
inv_arr = np.linalg.inv(arr)
print("Inversa de la matriz:")
print(inv_arr)