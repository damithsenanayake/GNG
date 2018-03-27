import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve,make_blobs,load_digits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#
X, colors = make_swiss_roll(n_samples=1000,random_state=10)#oad_digits()#make_blobs(n_samples=1000, n_features=3, centers=4, random_state=5)#
#
# data = load_digits()

# X = data.data
# X/=X.max()
# colors = data.target


W = np.random.random((3, X.shape[1]))
Y = np.random.random((3, 2))

G = np.zeros((W.shape[0], W.shape[0]))
errors = np.zeros(3)
a_max =1
ages = np.zeros(G.shape)

hits = np.zeros(G.shape[0])

lrst = 0.1
alpha = 0.6
maxiter = 20
for i in range(maxiter):

    for x in X:
        print '\r iteration : ', i, ' : n(G) : ', G.shape[0],
        dists = np.linalg.norm(W - x, axis=1)
        s, t, u = np.argsort(dists)[:3]
        lr = np.exp(0.5*i**2*1./maxiter**2) * lrst
        try:
            errors[s] += dists[s] ** 2
        except:
            np.append(errors, dists[s] ** 2)

        neis = np.where(G[s])[0]
        D = dists[neis]
        d = np.linalg.norm(Y[neis]-Y[s], axis=1)
        if d.shape[0] and d.max():
            d/= d.max()

        lamb = np.nan_to_num(np.array([np.exp(-.5*d**2)]).T)
        if lamb.sum() :
            lamb /= lamb.sum()
        lamb = np.nan_to_num(lamb)
        W[neis] += lr * (x - W[neis])*lamb
        W[s] += lr * (x - W[s])
        hits[s]+=1

        # Move Y


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


        if i % 8 == 0 and i < 9000:
            if errors[s] > 50.1:
                try:
                    ninds = np.where(G[s]==1)[0]
                    l, m = np.argsort(errors[ninds])[:3]
                except ValueError:
                    continue
                W_n = W[ninds[l]] + W[s] + W[ninds[m]]
                W_n /= 3.

                W_n = 2* W[s] - W_n

                Y_n = Y[ninds[l]] + Y[s] + Y[ninds[m]]
                Y_n /= 3.

                Y_n = 2*Y[s] - Y_n

                errors[l] *= alpha
                errors[s] *= alpha
                W = np.concatenate((W, np.array([W_n])), axis=0)
                Y = np.concatenate((Y, np.array([Y_n])), axis=0)
                G = np.concatenate((G, np.array([np.zeros(G.shape[0])])), axis=0)
                G = np.concatenate((G, np.array([np.zeros(G.shape[0])]).T), axis=1)
                G[s][-1]=1
                G[-1][s]=1
                ages = np.concatenate((ages, np.array([np.zeros(ages.shape[0])])), axis=0)
                ages = np.concatenate((ages, np.array([np.zeros(ages.shape[0])]).T), axis=1)
                errors = np.append(errors, 0)
                hits = np.append(hits, 0)

                #move y
                for _ in range(50):
                    for p in range(Y.shape[0]):
                        y_neis = np.where(G[p] == 1)[0]

                        oths = np.where(G[p] == 0)[0]  # np.array(range(G.shape[0]))#

                        d_oths = np.linalg.norm(Y[oths] - Y[p], axis=1)

                        pushdirs = np.array([np.exp(-d_oths)]).T  * 100
                        # pushdirs /= pushdirs.min()

                        push = (Y[oths] - Y[p]) * pushdirs

                        Y[oths] += push

                        Y[p] += lr * (Y[y_neis] - Y[p]).sum(axis=0)


    emptyNodes = np.where((G.sum(axis=0) <= 1))# | ( hits==0))

    W = np.delete(W, emptyNodes, axis=0)
    Y = np.delete(Y, emptyNodes, axis=0)
    G = np.delete(G, emptyNodes, axis=0)
    G = np.delete(G, emptyNodes, axis=1)
    ages = np.delete(ages,emptyNodes,axis=0)
    ages = np.delete(ages,emptyNodes,axis=1)
    errors = np.delete(errors, emptyNodes)
    hits = np.delete(hits, emptyNodes)

# Map Correction
print '\n'
for _ in range(1000):
    for p in range(Y.shape[0]):
        print '\rcorrecting:, ',_,
        y_neis = np.where(G[p] == 1)[0]

        oths = np.array(range(G.shape[0]))#np.where(G[p] == 0)[0]

        d_oths = np.linalg.norm(Y[oths] - Y[p], axis=1)
        # d_oths = d_oths/d_oths.sum()
        pushdirs = np.array([np.exp(-d_oths)]).T * 50


        push = (Y[oths] - Y[p])* pushdirs

        Y[oths] += push


        Y[p] += 0.01*(Y[y_neis]-Y[p]).sum(axis=0)
# print ages.max()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.T[0], X.T[1],X.T[2], c=colors, cmap=plt.cm.hsv, alpha=0.2)
ax.scatter(W.T[0], W.T[1],W.T[2], c='black', alpha=0.8)


for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if G[i, j]:
            ax.plot([W[i, 0], W[j, 0]], [W[i, 1], W[j, 1]],[W[i,2], W[j,2]], c='black')

plt.show()
# # #
plt.scatter(Y.T[0], Y.T[1], c='black', alpha = 0.4, s=4)

for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if G[i, j]:
            plt.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], c='black')
plt.show()
# #
predictions =[]

for x in X:
    predictions.append(Y[np.argmin(np.linalg.norm(W-x, axis=1))])

disp = np.array(predictions)

plt.scatter(disp.T[0], disp.T[1], c=colors, cmap=plt.cm.hsv, alpha = 0.1, s=4)
plt.show()
