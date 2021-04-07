import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create data
plt.figure(1)
noise = np.random.normal(0, 1, 5)
data = np.array([(0.0, 3.85), (2.5, 3.90), (5.0, 8.40), (7.5, 12.83), (10.0, 16.27)])
X = np.resize(data[:, 0], (5, 1))
Y = np.resize(data[:, 1], (5, 1))
plt.scatter(X, Y, marker='+')
plt.title('Linear data')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Theta
theta = np.resize([2, 1.5], (1, 2))

# Plot model
X_ = np.hstack((np.ones((5, 1)), X))
h = np.dot(X_, np.transpose(theta))
plt.plot(X, h, label='y=' + str(theta[0][1]) + 'x+' + str(theta[0][0]) + '', color='r')
plt.legend()
plt.show()

h = np.dot(X_, np.transpose(theta))
J_ = sum((Y - h)**2)


J = []
m_array = np.linspace(-3, 6, 100)
for m in m_array:
    theta = np.resize([2, m], (1, 2))
    h = np.dot(X_, np.transpose(theta))
    J.append(sum((Y - h)**2))

plt.plot(m_array, J, color='b')
plt.scatter(1.5, J_, c='r')
plt.title('Cost function')
plt.xlabel("m")
plt.ylabel("E")

J = []
b_array = np.linspace(-3, 6, 100)
for b in b_array:
    theta = np.resize([b, 1.5], (1, 2))
    h = np.dot(X_, np.transpose(theta))
    J.append(sum((Y - h)**2))

plt.plot(m_array, J, color='b')
plt.scatter(1.5, J_, c='r')



def error(m, b, ):
    theta = np.resize([b, m], (1, 2))
    h = np.dot(X_, np.transpose(theta))
    return sum((Y - h) ** 2)
    
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
M, B = np.meshgrid(m_array, b_array)
zs = np.array([error(mp, bp) for mp, bp in zip(np.ravel(M), np.ravel(B))])
Z = zs.reshape(M.shape)
ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.5)
ax.scatter(1.5, 2, 8.7643, 'r.')
ax.set_title('Cost function')
ax.set_xlabel("m")
ax.set_ylabel("b")
ax.set_zlabel("E")
plt.show()
