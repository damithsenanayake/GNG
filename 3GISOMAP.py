import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve, make_blobs, load_digits, fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D


# plt.style.use('dark_background')
def t_dist(d, n=1.5):
    dists = np.power((1 + d ** 2), -n)
    return dists / dists.sum()

def frw(W, G):
    print '\r running frw',
    D = np.zeros(G.shape)
    D.fill(np.inf)
    n = G.shape[0]
    for i in range(n):
        for j in range(n):
            if G[i][j]:
                D[i][j] = np.linalg.norm(W[i]-W[j])
            elif i==j:
                D[i][j] = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i][j] > D[i][k] + D[k][j]:
                    D[i][j] = D[i][k] + D[k][j]
    return np.nan_to_num(D)
#
# X, colors = make_swiss_roll(n_samples=1000,random_state=10)#oad_digits()#make_blobs(n_samples=1000, n_features=3, centers=4, random_state=5)#
#
data = load_digits()

X = data.data
colors = data.target

intdim = 3

g_max = 10000

W = np.random.random((intdim, X.shape[1]))

G = np.zeros((W.shape[0], W.shape[0]))
errors = np.zeros(intdim)
gens = np.zeros(intdim)
a_max_st = 3
ages = np.zeros(G.shape)

hits = np.zeros(G.shape[0])

lrst = 0.9
alpha = 0.01
maxiter = 100
D = X.shape[1]

QE = []
NG = []
GTs = []
print 'Graph Dimensionality : ', intdim
for i in range(maxiter):
    a_max = 2 # -(i%2==0)# (a_max_st*(1 - i * 1. / maxiter))
    wd = 0.0025
    sf = 0.1
    GT = -np.log(sf) * X.shape[1] * np.exp(-7.5 * (1. * i) **6 / maxiter ** 6)
    GTs.append(GT)
    QE.append(errors.sum())
    NG.append(G.shape[0])
    struct_change = (i % 9 == 0)
    errors.fill(0)
    for x in X:
        print '\r iteration : ', i, ' : n(G) : ', G.shape[0],
        dists = np.linalg.norm(W - x, axis=1)
        candidates = np.argsort(dists)[:intdim]

        #
        # if i%4 == 0:
        # t, u = np.argsort(np.linalg.norm(W-W[s], axis=1))[1:3]
        s = candidates[0]
        lr = np.exp(0.5 * i ** 2 * 1. / maxiter ** 2) * lrst
        try:
            errors[s] += dists[s] ** 2
        except:
            np.append(errors, dists[s] ** 2)

        neis = np.where(G[s])[0]
        D = dists[neis]
        d = np.linalg.norm(W[neis] - W[s], axis=1)
        if d.shape[0] and d.max():
            d /= d.max()

        lamb = np.nan_to_num(np.array([np.exp(-.5 * d ** 2)]).T)
        if lamb.sum():
            lamb /= lamb.sum()
        lamb = np.nan_to_num(lamb)
        print ' moving node',
        W[s] += lr * (x - W[s])
        W[neis] += lr * lamb * (x - W[neis])  - wd*W[neis]*(1-np.exp(-2.5*i/maxiter))
        hits[s] += 1

        # Move Y

        ages[s] += G[s]
        ages[s][s] = 0
        # if not( i%4 == 0) :

        G[s][ages[s] >= a_max] = 0
        G[:, s][ages[s] >= a_max] = 0
        ages[s][ages[s] >= a_max] = 0
        ages[:, s][ages[s] >= a_max] = 0

        for t in candidates[1:]:
            G[s][t] = 1
            G[t][s] = 1
            ages[s][t] = 0
            ages[t][s] = 0

            for u in candidates[1:]:
                if not (u == t):
                    G[u][t] = 1
                    G[t][u] = 1

                    ages[u][t] = 0
                    ages[t][u] = 0

    if struct_change:
        print ' creating new node, ',
        while errors.max() > GT **2 :
            grix = np.where(errors > GT ** 2)[0]
            for s in range(G.shape[0]):
                if errors[s] > GT ** 2:  # and gens[s] < g_max:
                    try:
                        ninds = np.where(G[s] == 1)[0]
                        h_er_ixs = ninds[np.argsort(errors[ninds])][:intdim]
                    except ValueError:
                        continue
                    W_n = np.sum(W[h_er_ixs], axis=0) + W[s]
                    W_n /= intdim * 1.

                    W_n = 2 * W[s] - W_n

                    errors[h_er_ixs] *= alpha
                    errors[s] *= alpha
                    gens[s] += 1
                    gens[h_er_ixs] += 1
                    W = np.concatenate((W, np.array([W_n])), axis=0)
                    G = np.concatenate((G, np.array([np.zeros(G.shape[0])])), axis=0)
                    G = np.concatenate((G, np.array([np.zeros(G.shape[0])]).T), axis=1)
                    G[s][-1] = 1
                    G[-1][s] = 1
                    G[h_er_ixs][:, -1] = 1
                    G[:, -1][h_er_ixs] = 1
                    ages = np.concatenate((ages, np.array([np.zeros(ages.shape[0])])), axis=0)
                    ages = np.concatenate((ages, np.array([np.zeros(ages.shape[0])]).T), axis=1)

                    errors = np.append(errors, errors[s])
                    hits = np.append(hits, 0)
                    gens = np.append(gens, 0)

                # move y
    move_range = 10
    if struct_change: move_range = 50*(i**2/maxiter**2)
    if i +1 >= maxiter: move_range = 200
    print ' moving all nodes in graph, ',


    if struct_change or i == maxiter - 1:
        emptyNodes = np.where((G.sum(axis=0) <= intdim - 2))  # | ( hits<=1))
        while emptyNodes[0].shape[0]:
            W = np.delete(W, emptyNodes, axis=0)
            G = np.delete(G, emptyNodes, axis=0)
            G = np.delete(G, emptyNodes, axis=1)
            ages = np.delete(ages, emptyNodes, axis=0)
            ages = np.delete(ages, emptyNodes, axis=1)
            errors = np.delete(errors, emptyNodes)
            hits = np.delete(hits, emptyNodes)
            hits.fill(0)
            emptyNodes = np.where((G.sum(axis=0) <= intdim - 2))  # | ( hits<=1))

# smoothiter = 50
# for i in range(smoothiter):
#     for x in X:
#         b = np.argmin(np.linalg.norm(x - W, axis=1))
#
#         neis = np.where(G[b])[0]
#
#         dists = np.linalg.norm(Y[b] - Y[neis], axis=1)
#         if dists.max():
#             dists/=dists.max()
#         lamb = np.exp(- dists **2)
#         W[b] += (x -W[b]) * 0.1
#         W[neis] += (x - W[neis]) * 0.1 * np.array([lamb]).T

        # Y[neis] += (Y[b]- Y[neis])*0.01


pwd = frw(W, G)

Y = MDS(dissimilarity='precomputed').fit_transform(pwd)
# # #
fig1 = plt.figure()

plt.scatter(Y.T[0], Y.T[1], c='black', alpha = 0.4, s=4)
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if G[i, j]:
            plt.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], c='grey')
plt.show(block=False)
# #
fig2 = plt.figure()
predictions = []

for x in X:
    predictions.append(Y[np.argmin(np.linalg.norm(W - x, axis=1))])

disp = np.array(predictions)

plt.scatter(disp.T[0], disp.T[1], c=colors, cmap=plt.cm.hsv, alpha=0.2, s=16, edgecolors=None)
plt.show(block=False)
fig3 = plt.figure()
t = range(len(QE))
plt.plot(t, QE)
plt.show()
fig4 = plt.figure()
plt.plot(t, GTs)
plt.show(block=False)
fig5 = plt.figure()
plt.plot(t, NG)
plt.show(block=False)
