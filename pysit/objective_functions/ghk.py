import numpy as np
from numpy import linalg as LA

import scipy
from scipy.stats import norm


def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    epsl = 0.01
    s1 = epsl*np.log(1 + np.exp(inputs / epsl))
    ds1 = np.exp(inputs / epsl) / (1 + np.exp(inputs / epsl))
    s2 = s1 + epsl*np.log(1 + np.exp(-inputs / epsl))
    ds2 = np.exp(inputs / epsl) / (1 + np.exp(inputs / epsl)) - (np.exp(-inputs / epsl) / (1 + np.exp(-inputs / epsl)))
    return s2, ds2


def GHK(P, Q, epsilon, lamb1, lamb2, niter, Ct, Cx, dt, dx):

    eps = 2.2204e-16
    ep = eps**6
    dx2 = dx**2
    dt2 = dt**2

    def div0(x, y, pw):
        # works for 0 = 0 *x  return element by element
        r = np.zeros(x.flatten().shape)
        II = np.where(((x.flatten()) >= 0) & ((y.flatten()) > 0))
        r[II] = (np.divide(x.flatten()[II], y.flatten()[II]))**pw
        r = np.reshape(r,(x.shape))
        return r

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Ct = np.asarray(Ct, dtype=np.float64)
    Cx = np.asarray(Cx, dtype=np.float64)

    # utilities
    #zp = np.zeros(P.shape)
    op = np.ones(P.shape)
    #zq = np.zeros(Q.shape)
    oq = np.ones(Q.shape)

    le1 = lamb1 + epsilon
    pw1 = lamb1/le1
    le2 = lamb2 + epsilon
    pw2 = lamb2/le2

    Kt = np.exp(-Ct/epsilon)
    Kx = np.exp(-Cx/epsilon)

    # init
    a = op
    b = oq  # a is column and b is line

    cvrgce = np.zeros((1, niter))  # errors
    icv = 1  # iteration for errors

    for i in range(niter):
        a0 = a  # b0 = b

        # sinkhorn iterates here
        #a = (div0(P, dt*dx*(np.dot(Kt.dot(b), Kx)), zp)**pw1)
        #b = (div0(Q, dt*dx*(np.dot(Kt.dot(a), Kx)), zq)**pw2)
        a = div0(P, dt*dx*(np.dot(Kt.dot(b), Kx)), pw1)
        b = div0(Q, dt*dx*(np.dot(Kt.dot(a), Kx)), pw2)

        if i % 10 == 0:
            Ig = np.where(((a.flatten()) > ep) & ((a0.flatten()) > ep))
            #err = np.linalg.norm((np.log(a.flatten()[Ig])-np.log(a0.flatten()[Ig])), np.inf)
            er = (np.log(a.flatten()[Ig])-np.log(a0.flatten()[Ig]))
            err = LA.norm(er, np.inf)
            cvrgce[0, icv] = err
            icv = icv + 1

    conv = cvrgce[0, icv-1]
    gamma = a*(np.dot(Kt.dot(b), Kx))

    Aat = np.zeros(a.flatten().shape)
    Bbt = np.zeros(b.flatten().shape)
    IA = np.where((a.flatten()) > 0)
    IB = np.where((b.flatten()) > 0)

    Aat[IA] = -lamb1*dx*dt*(a.flatten()[IA]**(-epsilon/lamb1) - 1)
    Bbt[IB] = -lamb2*dx*dt*(b.flatten()[IB]**(-epsilon/lamb2) - 1)

    At = np.reshape(Aat, (a.shape))
    Bt = np.reshape(Bbt, (b.shape))

    #J = np.sum(P.dot(At.T) + Bt.dot(Q.T) - epsilon*dt2*dx2*gamma)
    J = np.sum(P*At + Bt*Q - epsilon*dt2*dx2*gamma)

    return J, Bt, conv

