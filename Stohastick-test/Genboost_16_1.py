import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)

# https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html
# Сгенерируем данные для задач классификации

# 3D
features_separate_3D, labels_separate_3D  = make_blobs(n_samples=1000,n_features=3, centers=[[1.,1.,1.],[-1.,-1.,-1.]])
labels_separate_3D = list(map(lambda x: 1 if x == 1 else -1,labels_separate_3D))
#features_separate_3D = list(features_separate_3D)

# 2D
features_separate_2D, labels_separate_2D  = make_blobs(n_samples=1000,n_features=2, centers=[[1.,1.],[-1.,-1.]])
labels_separate_2D = list(map(lambda x: 1 if x == 1 else -1,labels_separate_2D))
#features_separate_2D = list(features_separate_2D)

# 1D
features_separate_1D, labels_separate_1D  = make_blobs(n_samples=1000,n_features=1, centers=[[1.],[-1.]], cluster_std=0.4)
labels_separate_1D = list(map(lambda x: 1 if x == 1 else -1,labels_separate_1D))

# print(labels_separate_2D)
# print(features_separate_2D[1])
# fig = plt.figure(figsize=plt.figaspect(2.))
# ax = fig.add_subplot(211, projection='3d',xlabel='$X_1$',ylabel='$X_2$',zlabel='$X_3$')
# ax.scatter(features_separate_3D[:, 0], features_separate_3D[:, 1],features_separate_3D[:, 2], marker='o', c=labels_separate_3D)
# ax = fig.add_subplot(212, xlabel='$X_1$',ylabel='$X_2$')
# ax.scatter(features_separate_2D[:, 0], features_separate_2D[:, 1], marker='o', c=labels_separate_2D)
# plt.show()

# 3D визуализация
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d',xlabel='$X_1$',ylabel='$X_2$',zlabel='$X_3$')
# ax.scatter(features_separate_3D[:, 0], features_separate_3D[:, 1],features_separate_3D[:, 2], marker='o', c=labels_separate_3D)
# plt.show()

# 2D визуализация
# fig = plt.figure()
# ax = fig.add_subplot(111, xlabel='$X_1$',ylabel='$X_2$')
# ax.scatter(features_separate_2D[:, 0], features_separate_2D[:, 1], marker='o', c=labels_separate_2D)
# plt.show()

# 1D визуализация
# fig = plt.figure()
# ax = fig.add_subplot(111, xlabel='$X_1$',ylabel='$X_2$')
# ax.scatter(features_separate_1D[:],np.zeros(1000) ,marker='o', c=labels_separate_1D)
# ax.plot([x for x in [0:1000]],)
# plt.show()

# Плоскость в 3-хмерном пространстве
def hyperplane(x, y, A=1, B=1, C=1):
    return A*x +B*y + C

# Линейная дискриминантная дискриминантная функция
def f(features,weights):
    sum = weights[0]
    for x,w in zip(features,weights[1:]):
        sum += x*w
    return sum



# def sign(n, eps=0.000000001):
#     if n < -eps:
#         return -1
#     elif n > eps:
#         return 1
#     else:
#         return 0


# Параметризованное семейство алгоритмов
def a(x,w):
    return np.sign(f(x,w))


# Отступ (margin) объекта x от класса y на адлгоритме a
def M(x,y,w):
    return y * f(x, w)


#  Натация айверса для определения эмпирического риска [M < 0 ]
def Ayvers_M(x,y,w):
    if M(x,y,w) < 0:
        return 1
    else:
        return 0


# Эмпирический риск
def Q(w,features, labels):
    return sum([Ayvers_M(x,y,w) for x,y in zip(features, labels)])


# def func(_w):
#     def z(x,y):
#         return -1/_w[2] * (_w[0]*x + _w[1]*y + _w[3])
#     return z

# X = [x for x in range(-2, 2, 0.1)]
# Y = [y for y in range(-2, 2, 0.1)]
# xx, yy  = np.meshgrid(X,Y)
# print([sign(y) for y in range(-5,5,1)])

# plt.plot([x for x in range(-5,5,1)], [sign(y) for y in range(-5,5,1)], "ro")
# plt.show()

X,Y = np.meshgrid(np.linspace(-5,5,2),np.linspace(-5,5,2))
Z = hyperplane(X,Y,A=2,B=2, C=2)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d',xlabel='$X_1$',ylabel='$X_2$',zlabel='$X_3$',xlim=(-5.,5.),ylim=(-5.,5.),zlim=(-5.,5.))
# ax.scatter(features_separate_3D[:, 0], features_separate_3D[:, 1],features_separate_3D[:, 2], marker='o', c=labels_separate_3D)
# ax.plot_surface(X,Y,Z, alpha =0.2)
# plt.show()

weights_1D = [0., -1.]
weights_2D = [1., 1.]
weights_3D = [1., 1. , 1.]
x = np.arange(-1,1,0.1)
y = x  

# fig = plt.figure()
# ax = fig.add_subplot(111, xlabel='$X_1$',ylabel='$X_2$', ylim=(-1.,1.))
# ax.scatter(features_separate_1D[:],np.zeros(1000) ,marker='o', c=labels_separate_1D)
# ax.plot(x,y)
# plt.show()

print(f(features_separate_1D[1], weights_1D))
print(M(features_separate_1D[1], labels_separate_1D[1], weights_1D))
print(labels_separate_1D[1])
print(Ayvers_M(features_separate_1D[1], labels_separate_1D[1], weights_1D))
print(Q(weights_1D, features_separate_1D, labels_separate_1D))

Grid_1D_1 = np.array([Q([x,-1],features_separate_1D,labels_separate_1D) for x in np.arange(-1.1,1.1,0.001) ])
print(Grid_1D_1)

Grid_1D_2 = np.array([Q([x,1],features_separate_1D,labels_separate_1D) for x in np.arange(-1.1,1.1,0.001) ])
print(Grid_1D_2)
