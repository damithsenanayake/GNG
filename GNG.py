import numpy as np
from sklearn.datasets import make_blobs,make_circles, make_s_curve, make_moons
import matplotlib.pyplot as plt

g_size = 10

G = np.zeros(shape=(g_size, g_size))#np.identity(g_size)

ages = np.zeros(shape=G.shape)

errors = np.zeros(g_size)

W = np.random.random(size=(g_size,2))

X, c = make_blobs(n_samples= 500, centers= 5, n_features=2, random_state=1, cluster_std=0.5)     #make_moons(n_samples=500)#
## Train Neural Gas

X -= X.min()
X /= X.max()

alpha = 0.1
a_max = 100

for i in range(10000):

    for x in X:
        dists = np.linalg.norm(W-x, axis=1)
        s, t = np.argsort(dists)[:2]

        try:
            errors[s] += dists[s]**2
        except:
            np.append(errors, dists[s]**2)



        neis = np.where(G[s])[0]

        d = dists[neis]
        d /= (d.sum()+0.00001)
        W[neis] += 0.01*(x-W[neis])#*np.array([np.exp(-0.5*d**2)]).T

        ages[s]+=G[s]

        ages[s][s]=0

        G[s][ages[s]>a_max]=0
        ages[s][ages[s]>a_max]=0

        if G[s][t]:
            ages[s, t] =0
        else:
            G[s][t] = 1
            ages[s][t]=0



        if i%50 ==48 and i < 9000:
            if errors[s] > 1:
                try:
                    l = np.argmax(errors[G[s]==1])
                except IndexError:
                    print errors
                W_n = W[l] + W[s]
                W_n *= 0.5
                errors[l]*= alpha
                errors[s]*= alpha
                W=np.concatenate((W, np.array([W_n])), axis=0)

                G = np.concatenate((G, np.array([np.zeros(G.shape[0])])), axis=0)
                G = np.concatenate((G, np.array([np.zeros(G.shape[0])]).T), axis=1)
                ages = np.concatenate((ages, np.array([np.zeros(ages.shape[0])])), axis=0)
                ages = np.concatenate((ages, np.array([np.zeros(ages.shape[0])]).T), axis=1)
                errors = np.append(errors, 0)


    GT = G
    agesT = ages
    errorsT = errors
    WT = W
    for s in range(G.shape[0]):
        if G[s].sum() == 0:
                    WT=np.delete(W, s, 0)
                    GT=np.delete(G, s, 0)
                    GT=np.delete(GT, s, 1)
                    agesT=np.delete(ages, s, 0)
                    agesT=np.delete(agesT, s, 1)
                    errorsT = np.delete(errors, s)

    G = GT
    ages = agesT
    errors = errorsT
    W = WT
    
# print ages

#### Visualize Neural Gas
plt.scatter(X.T[0], X.T[1], c=c, cmap=plt.cm.Set1, alpha=0.2)
plt.scatter(W.T[0], W.T[1], c='black', alpha=0.2)

for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if G[i,j]:
            plt.plot([W[i,0], W[j, 0]], [W[i,1],W[j, 1]], c='black')

plt.show()