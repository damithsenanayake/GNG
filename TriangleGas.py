import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_s_curve, make_moons, make_swiss_roll
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import pickle

g_size = 3

G = np.zeros(shape=(g_size, g_size))  # np.identity(g_size)

ages = np.zeros(shape=G.shape)

errors = np.zeros(g_size)

W = np.random.random(size=(g_size, 3))

Y = np.random.random(size=(g_size,2))

X, c = make_blobs(500, random_state=10, n_features=3)#make_s_curve(n_samples=500, random_state=10)#make_blobs(n_samples=1000, centers=5, n_features=2, random_state=1,
                  #cluster_std=0.5)  # make_moons(n_samples=500)#
## Train Neural Gas

X -= X.min()
X /= X.max()
mds = MDS()
alpha = 0.1
a_max = 2

for i in range(1000):

    for x in X:
        dists = np.linalg.norm(W - x, axis=1)
        s, t, u = np.argsort(dists)[:3]

        try:
            errors[s] += dists[s] ** 2
        except:
            np.append(errors, dists[s] ** 2)

        neis = np.where(G[s])[0]
        d = dists[neis]
        d /= (d.sum() + 0.00001)
        W[neis] += 0.01 * (x - W[neis])  # *np.array([np.exp(-0.5*d**2)]).T
        W[s] += 0.1 * (x - W[s])

        # Move Y

        if i%20 == 10:
            for p in range(Y.shape[0]):
                y_neis = np.where(G[p] == 1)[0]

                d = np.linalg.norm(Y[y_neis] - Y[p], axis=1)
                D = np.linalg.norm(W[y_neis] - W[p], axis=1)

                # print d-D

                dirs = d - D  # d/d.sum() - D/D.sum()
                dirs = np.array([dirs]).T

                Y[y_neis] += dirs * (Y[p] - Y[y_neis])


        ages[s] += G[s]
        ages[s][s] = 0

        G[s][ages[s] > a_max] = 0
        G[:, s][ages[s] > a_max] =0
        ages[s][ages[s] > a_max] = 0
        ages[:, s][ages[s] > a_max] =0
        if G[s][t]:
            ages[s, t] = 0
            ages[t, s] = 0
        else:
            G[s][t] = 1
            G[t][s] = 1
            ages[s][t] = 0
            ages[t][s] =0

        if not G[s][u]:
            G[s][u]=1
            G[u][s]=1
        ages[s][u]=0
        ages[u][s]=0

        if not G[u][t]:
            G[u][t]=1
            G[t][u]=1

        ages[u][t]=0
        ages[t][u]=0


        if i % 50 == 48 and i < 9000:
            if errors[s] > 0.2:
                try:
                    ninds = np.where(G[s]==1)[0]
                    l, m = np.argsort(errors[ninds])[:3]
                except ValueError:
                    continue
                W_n = W[ninds[l]] + W[s] + W[ninds[m]]
                W_n /= 3.


                errors[l] *= alpha
                errors[s] *= alpha
                W = np.concatenate((W, np.array([W_n])), axis=0)
                G = np.concatenate((G, np.array([np.zeros(G.shape[0])])), axis=0)
                G = np.concatenate((G, np.array([np.zeros(G.shape[0])]).T), axis=1)
                ages = np.concatenate((ages, np.array([np.zeros(ages.shape[0])])), axis=0)
                ages = np.concatenate((ages, np.array([np.zeros(ages.shape[0])]).T), axis=1)
                errors = np.append(errors, 0)


    emptyNodes = np.where(G.sum(axis=0)<=1)

    W = np.delete(W, emptyNodes, axis=0)
    G = np.delete(G, emptyNodes, axis=0)
    G = np.delete(G, emptyNodes, axis=1)
    ages = np.delete(ages,emptyNodes,axis=0)
    ages = np.delete(ages,emptyNodes,axis=1)
    errors = np.delete(errors, emptyNodes)



#Projection step

Y = np.random.uniform(size=(G.shape[0], 2))

# print len(np.where(G.sum(axis=1)==0)[0])
#
# for i in range(0):
#     Y -= Y.min()
#     Y /= Y.max()
#     for k in range(W.shape[0]):
#         b = W[k]
#         if i < 100000:
#             neis = np.argsort(np.linalg.norm(W[k] - W, axis=1))[:5]
#         else:
#             neis = np.where(G[k]==1)[0]#np.argsort(np.linalg.norm(W[k]-W, axis=1))[:5]
#
#         D = np.linalg.norm(W[k]-W[neis], axis=1)
#
#         d = np.linalg.norm(Y[k] - Y[neis], axis=1)
#
#         D/=D.sum()
#         d/=d.sum()
#         dirs = d-D
#
#         Y[neis] += 0.2*(np.exp(-0.5*i/20000.))* (Y[k]- Y[neis]) * np.array([dirs]).T

# print ages

#### Visualize Neural Gas

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.T[0], X.T[1],X.T[2], c=c, cmap=plt.cm.hsv, alpha=0.2)
ax.scatter(W.T[0], W.T[1],W.T[2], c='black', alpha=0.2)


for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if G[i, j]:
            ax.plot([W[i, 0], W[j, 0]], [W[i, 1], W[j, 1]],[W[i,2], W[j,2]], c='black')



plt.show()

pickle.dump(fig, open('FigureObject.fig.pickle','wb'))
#
# plt.scatter(Y.T[0], Y.T[1], c='black', alpha = 0.4)
#
# for i in range(G.shape[0]):
#     for j in range(G.shape[1]):
#         if G[i, j]:
#             plt.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], c='black')
# plt.show()
