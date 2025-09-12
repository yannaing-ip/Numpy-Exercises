import numpy as np

v1 = np.array([[1, 2], [3, 2]])
v2 = np.array([[3, 1],[2, 6]])

#Get Size of v1
print(f"Total item of v1 => {v1.size}")

#Shape of v1
print(f"Dimension of v1 => {v1.shape}")

#Scalar product
print(2 * v1)

#Addiction of two vector
print(v1 + v2)

#Matrix Multiplication
print(np.dot(v1, v2))

#Get data type
print(f"Data type of v1 => {v1.dtype}")

#Creating 2x3 vector of dtype=int16
v3 = np.array([[2, 1, 4],[2, 2, 3]], dtype="int16")
print(v3)

#Reshaping v3 to 3x2
print(np.reshape(v3, (3,2)))

#Transpose of v3
print(np.transpose(v3))

print(v3.nbytes)

v4 = np.random.rand(3,3)
print(v4)

#Inverse of v4
try:
	print(np.linalg.inv(v4))
except np.linalg.LinAlgError:
	print("Matrix is not unvertiable")

#Deteminant of v4
print(np.linalg.det(v4))