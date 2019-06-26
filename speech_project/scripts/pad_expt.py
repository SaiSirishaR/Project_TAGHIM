import numpy
import numpy as np

A = np.array([[1,2],[3,4]])
print("A is", A, "shape is", numpy.shape(A))
B = np.pad(A, ((2,0),(1,1)), 'constant')
print("a is", B, "shape is", numpy.shape(B))
