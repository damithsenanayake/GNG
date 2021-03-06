import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X, t = make_s_curve(n_samples=500, random_state=100)#make_swiss_roll(n_samples=500,random_state=10)

W = np.random.random((3, X.shape[1]))
Y = np.random.random((3, 2))

G = np.zeros((W.shape[0], W.shape[0]))
errors = np.zeros(3)
amax =2
ages = np.zeros(G.shape)

lrst = 0.5
alpha = 0.2
for i in range(2000):
    print '\riteration ', i+1, ' n(G) = ', G.shape[0],
    for x in X:
        lr = lrst*np.exp(-.5*i**2/50.**2)
        k = np.argmin(np.linalg.norm(W-x, axis=1))
        l, m = np.argsort(np.linalg.norm(W[k]-W, axis=1))[1:3]

        # hneis = np.argsort(np.linalg.norm(x-W, axis=1))[1:3]


        # G[l][m]=1
        # G[m][l]=1
        # if not np.in1d(hneis, l).any() :
        #     G[k][l]=0
        #     G[l][k]=0
        #
        # if not np.in1d(hneis, m).any() :
        #     G[k][m] =0
        #     G[m][k]=0
        #

        # ages[l][m] = 0
        # ages[m][l] = 0

        err = np.linalg.norm(x-W[k])
        W[k] += (x - W[k])*lr

        neis = np.where(G[k] == 1)[0]
        non_neis = np.array(range(Y.shape[0]))#np.setdiff1d(np.array(range(Y.shape[0])), neis)#
        G[k][l] = 1
        G[l][k] = 1
        G[m][k] = 1
        G[k][m] = 1
        ages[k][neis] +=1
        ages[neis][:,k] +=1

        # if np.in1d(hneis, l).any():
        ages[k][l] = 0
        ages[l][k] = 0
        # # if np.in1d(hneis, m).any():
        ages[m][k] = 0
        ages[k][m] = 0
        # ages[k][hneis]=0
        # ages[:, k][hneis]=0

        d = np.linalg.norm(W[k]-W[neis], axis=1)
        d_n = np.linalg.norm(W[k]-W[non_neis], axis=1)

        # d /= d.max()
        # d_n /= d_n.max()

        theta = np.nan_to_num(np.array([np.exp(-0.5*d**2)]).T)
        theta = np.nan_to_num(theta/theta.sum())

        # if np.isinf(1./d_n).any() or np.isnan(1./d_n).any():
        #     print 'hold1'

        theta_n = np.nan_to_num(np.array([1./d_n**2]).T)#np.nan_to_num(np.array([np.exp(-.5*d_n**2)]).T)
        theta_n = np.nan_to_num(theta_n/theta_n.max())
        push = (Y[k]-Y[non_neis]) * theta_n

        W[neis] += (W[k]-W[neis])*lr#*theta



        Dw = np.linalg.norm(W[neis]-W[k], axis=1)
        dy = np.linalg.norm(Y[neis]-Y[k], axis=1)

        dirs = dy - Dw
        pull = np.array([dirs]).T * (Y[k]-Y[neis]) * lr# * theta

        if np.isnan(pull).any() or np.isinf(pull).any():
            print 'hold'
            continue

        Y[neis] += pull


        # Y[neis] += (Y[k]-Y[neis])*lr*theta
        if np.isnan(push).any() or np.isinf(push).any():
            print 'hold'
            continue
        Y[non_neis] -= push

        errors[k] += err**2

        ### Growth ####

        if errors[k] >=1 and i%2==0:
            errors[k]*= alpha
            errors[neis] += errors[neis]*alpha

            W_n = W[l] + W[m] + W[k]
            W_n /= 3.
            W_n = 2*W[k] - W_n

            Y_n = Y[l] + Y[m] + Y[k]
            Y_n /= 3.

            Y_n = 2*Y[k] - Y_n

            W = np.concatenate((W,np.array([W_n])), axis=0)
            Y = np.concatenate((Y, np.array([Y_n])), axis=0)

            errors = np.concatenate((errors, np.array([0])))

            G = np.concatenate((G, np.zeros((G.shape[0], 1))), axis=1)
            G = np.concatenate((G, np.zeros((G.shape[1], 1)).T), axis=0)

            ages = np.concatenate((ages, np.zeros((ages.shape[0], 1))), axis=1)
            ages = np.concatenate((ages, np.zeros((ages.shape[1], 1)).T), axis=0)


        ### edge deletion ###

        G[k][ages[k] >= amax] = 0
        ages[k][ages[k] >= amax] = 0
        G[:, k][ages[:,k] >= amax] = 0
        ages[:, k][ages[:, k] >= amax] = 0

    ### node deletion ###

    delcands = np.where(G.sum(axis=1)<=1)[0]

    ages = np.delete(ages, delcands, axis=0)
    ages = np.delete(ages,delcands, axis=1)
    G = np.delete(G, delcands, axis=0)
    G= np.delete(G, delcands, axis=1)
    Y = np.delete(Y, delcands, axis=0)
    W = np.delete(W, delcands, axis=0)
    errors = np.delete(errors, delcands)


print ages.max()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.T[0], X.T[1],X.T[2], c=t, cmap=plt.cm.hsv, alpha=0.2)
ax.scatter(W.T[0], W.T[1],W.T[2], c=range(W.shape[0]), cmap=plt.cm.hsv, alpha=0.8)


for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if G[i, j]:
            ax.plot([W[i, 0], W[j, 0]], [W[i, 1], W[j, 1]],[W[i,2], W[j,2]], c='black')

plt.show()

plt.scatter(Y.T[0], Y.T[1], c=range(Y.shape[0]), cmap= plt.cm.hsv, alpha = 0.4)

for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if G[i, j]:
            plt.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], c='black')
plt.show()

predictions =[]

for x in X:
    predictions.append(Y[np.argmin(np.linalg.norm(W-x, axis=1))])

disp = np.array(predictions)

plt.scatter(disp.T[0], disp.T[1], c=t, cmap=plt.cm.hsv, alpha = 0.4)
plt.show()