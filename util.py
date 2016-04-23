# -*- coding: utf-8 -*-
import cv2
import numpy as np


def ImageToPhysics(pixel, res):
    ''' convert pixel value in image coordinate to physical coordinate '''
    x = pixel[0] * res[0] # + origin.x
    y = pixel[1] * res[1] # + origin.y
    return (x, y)


def PhysicsToImage(phy, res):
    px = int(phy[0] / res[0])
    py = int(phy[1] / res[1])
    return (px, py)


'''
Curve smoothing
http://www.morethantechnical.com/2012/12/07/resampling-smoothing-and-interest-points-of-curves-via-css-in-opencv-w-code/
'''
def getGaussianDerivs(sigma, M):
    L = (M - 1)/2
    sigma_sq = sigma * sigma
    sigma_quad = sigma_sq * sigma_sq

    dg = np.zeros(M)
    d2g = np.zeros(M)

    gaussian = np.squeeze(cv2.getGaussianKernel(M, sigma, cv2.CV_64F))
    for i in range(-L, L+1):
        idx = i + L
        # from http://www.cedar.buffalo.edu/~srihari/CSE555/Normal2.pdf
        dg[idx] = (-i/sigma_sq) * gaussian[idx]
        d2g[idx] = (-sigma_sq + i*i)/sigma_quad * gaussian[idx]

    return gaussian, dg, d2g


def getdX(x, n, sigma, g, dg, d2g, is_open=True):
    L = (len(g) - 1) / 2
    gx = 0.0
    dgx = 0.0
    d2gx = 0.0
    for k in range(-L, L+1):
        if n - k < 0:
            if is_open:
                # open curve
                x_n_k = x[-(n-k)]
            else:
                # closed curve
                x_n_k = x[len(x) + (n-k)]
        elif n - k > len(x) - 1:
            if is_open:
                x_n_k = x[n+k]
            else:
                x_n_k = x[(n-k) - len(x)]
        else:
            x_n_k = x[n-k]
        gx += x_n_k * g[k + L]
        dgx += x_n_k * dg[k + L]
        d2gx += x_n_k * d2g[k + L]
    return gx, dgx, d2gx


def getdXcurve(x, sigma, g, dg, d2g, is_open=True):
    gx = np.zeros(len(x))
    dx = np.zeros(len(x))
    d2x = np.zeros(len(x))

    for i in range(len(x)):
        gausx, dgx, d2gx = getdX(x, i, sigma, g, dg, d2g, is_open)
        gx[i] = gausx
        dx[i] = dgx
        d2x[i] = d2gx

    return gx, dx, d2x

def smoothCurve(fit_x, fit_y):
    sigma = 3.0
    M = int(np.round((10.0*sigma + 1.0)/2.0) * 2 - 1)
    assert(M % 2 == 1)

    g, dg, d2g = getGaussianDerivs(sigma, M)
    new_fit_x, x, xx = getdXcurve(fit_x, sigma, g, dg, d2g)
    new_fit_y, y, yy = getdXcurve(fit_y, sigma, g, dg, d2g)

    # compute curvatures
    kappa = np.zeros(len(fit_x))
    for i in range(len(fit_x)):
        kappa[i] = (x[i] * yy[i] - xx[i] * y[i]) / (x[i]*x[i] + y[i]*y[i])**1.5

    return new_fit_x, new_fit_y, kappa


def arcLength(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def resampleCurve(x, y, N, is_open = True):
    pl_length = arcLength(x, y)
    resample_size = pl_length / N
    curr = 0
    dist = 0.0
    resample_x = np.zeros(N)
    resample_y = np.zeros(N)

    # set initial
    resample_x[0] = x[0]
    resample_y[0] = y[0]

    i = 1
    while i < N:
        assert(curr < len(x) - 1)
        last_dist = np.sqrt( (x[curr] - x[curr+1])**2 + (y[curr] - y[curr+1])**2 )
        dist += last_dist
        if dist > resample_size:
            _d = last_dist - (dist - resample_size)
            cp = np.array([x[curr], y[curr]])
            cp1 = np.array([x[curr+1], y[curr+1]])
            dirv = cp1 - cp
            dirv /= np.linalg.norm(dirv)
            ncp = cp + dirv * _d
            resample_x[i] = ncp[0]
            resample_y[i] = ncp[1]
            i += 1
            dist = last_dist - _d

            while dist - resample_size > 1e-3:
                resample_x[i] = resample_x[i - 1] + dirv[0] * resample_size
                resample_y[i] = resample_y[i - 1] + dirv[1] * resample_size
                dist -= resample_size
                i += 1
        curr += 1

    return resample_x, resample_y

def findCrossings(kappa):
    crossings = []
    for i in range(len(kappa) - 1):
        if (kappa[i] < 0 and kappa[i+1] > 0) or (kappa[i] > 0 and kappa[i+1] < 0):
            crossings.append(i)
    return np.array(crossings)