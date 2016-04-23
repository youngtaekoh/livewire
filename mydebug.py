# -*- coding:utf-8 -*-
import numpy as np
import util
from scipy import interpolate


class Curve:
    def __init__(self, x = None, y = None, sf = None):
        if (x is not None) and (y is not None):
            self.CreateByInterpolation(x, y, sf)

    def CreateByInterpolation(self, x, y, sf = None):
        """
        Create cubic spline by interpolating (x, y)

        :param x: x values
        :param y: y values
        :param sf: smoothing factor
        """
        # compute weights
        #dx, dy = np.diff([x, y])
        #weights = np.linspace(0.1, 0.9, len(x))
        #weights = weights / np.sum(weights)
        #print len(weights)

        if sf is None:
            self.tck, self.u = interpolate.splprep([x, y], k=4)
        else:
            print "smoothing factor:", sf
            self.smoothingFactor = sf
            self.tck, self.u = interpolate.splprep([x, y], k=4, s=self.smoothingFactor)

    def GetSamples(self, u = None):
        if u is None:
            u = self.u
        return interpolate.splev(u, self.tck)

    def GetCurvatureCurve(self):
        '''
        :return: approx. curvature curve (type of InterpolateUnivariateSpline)
        '''
        u_new = np.linspace(0, 1, len(self.u), endpoint=True)
        dx = interpolate.splev(u_new, self.tck, der=1)
        ddx = interpolate.splev(u_new, self.tck, der=2)
        # compute curvature samples
        k_samples = (dx[0] * ddx[1] - ddx[0] * dx[1]) / np.linalg.norm(dx) ** 3
        fs = interpolate.InterpolatedUnivariateSpline(u_new, k_samples, k=3)
        return fs

class DebugInfo:
    def __init__(self):
        self.points = None
        self.curves = None
        self.lastPt = None
        self.smoothing = None
        self.res = None
        # inflection points
        self.roots = None
        # interpolated output curve
        self.outcrv = None
        # fitting inputs (in image coordinate)
        self.imgFit_x = None
        self.imgFit_y = None

    def load(self, filename):
        self.points, self.curves, self.lastPt, self.smoothing, self.res = np.load(filename)

    def interpolate(self, fit_x, fit_y, smoothing = None, newu = None):
        if smoothing is not None:
            tck, u = interpolate.splprep([fit_x, fit_y], k=4, s=smoothing)
        else:
            tck, u = interpolate.splprep([fit_x, fit_y], k=4)

        if (newu is not None):
            u = newu

        out = interpolate.splev(u, tck)
        return out, u, tck

    def CurvesToFlat(self):
        fit_x = np.array([])
        fit_y = np.array([])
        for crv in self.curves:
            fit_x = np.hstack((fit_x, crv[:, 0]))
            fit_y = np.hstack((fit_y, crv[:, 1]))

        return (fit_x, fit_y)

    def ConvertToPhysicalCoord(self, fit_x, fit_y):
        # convert fit_x, fit_y into the physical coordinate.
        out_x = []
        out_y = []
        for i, (x, y) in enumerate(zip(fit_x, fit_y)):
            phy = util.ImageToPhysics((x, y), self.res)
            out_x.append(phy[0])
            out_y.append(phy[1])
        return (out_x, out_y)

    def curves_to_flat(self):
        fit_x = np.array([])
        fit_y = np.array([])
        for crv in self.curves:
            fit_x = np.hstack((fit_x, crv[:, 0]))
            fit_y = np.hstack((fit_y, crv[:, 1]))

        self.imgFit_x = fit_x.copy()
        self.imgFit_y = fit_y.copy()

        # convert fit_x, fit_y into the physical coordinate.
        for i, (x, y) in enumerate(zip(fit_x, fit_y)):
            phy = util.ImageToPhysics((x, y), self.res)
            fit_x[i] = phy[0]
            fit_y[i] = phy[1]

        try:
            tck, u = interpolate.splprep([fit_x, fit_y], k=4, s=self.smoothing)
            out = interpolate.splev(u, tck)

            # convert interpolated curve points to Image Coordinate.
            for i, (x, y) in enumerate(zip(out[0], out[1])):
                img = util.PhysicsToImage((x, y), self.res)
                out[0][i] = img[0]
                out[1][i] = img[1]

            self.outcrv = out
            self.interpCurve = np.column_stack((out[0], out[1])).astype(np.int32)

            # compute length
            dxdy = self.res * np.diff(self.interpCurve, axis=0)
            length = np.sum(np.linalg.norm(dxdy, 2, axis=1))

            unew = np.linspace(0, 1, len(u), endpoint=True)
            dx = interpolate.splev(unew, tck, der=1)
            ddx = interpolate.splev(unew, tck, der=2)
            # compute curvature samples
            k_sample = (dx[0] * ddx[1] - ddx[0] * dx[1]) / np.linalg.norm(dx) ** 3
            fs = interpolate.InterpolatedUnivariateSpline(unew, k_sample, k=3)
            roots = fs.roots()

            if (len(roots) > 0):
                outroots = []
                out = interpolate.splev(roots, tck)
                for ipt in np.transpose(out):
                    ipt = util.PhysicsToImage(ipt, self.res)
                    outroots.append(ipt)
                self.roots = outroots
            #self.inflections = roots
            #self.messagePrinted.emit("Inflection points: {} length: {:.2f} mm".format(len(roots), length))
            return (self.imgFit_x, self.imgFit_y, self.outcrv)

        except ValueError as e:
            print e
            return None

di = DebugInfo()
#di.load(r'D:\Documents\Projects\Python\livewire\liver\1.npy')
di.load(r'/Users/stardust/Projects/Python/livewire/liver/1.npy')
(input_x, input_y) = di.CurvesToFlat()
(input_x, input_y) = di.ConvertToPhysicalCoord(input_x, input_y)
new_fit_x, new_fit_y, kappa = util.smoothCurve(input_x, input_y)
res_x, res_y = util.resampleCurve(new_fit_x, new_fit_y, 100)

inputCrv = Curve(new_fit_x, new_fit_y, 1.0)
k_crv = inputCrv.GetCurvatureCurve()

import matplotlib.pyplot as plt

u = np.linspace(0, 1, len(inputCrv.u))
out = inputCrv.GetSamples(u)
k_out = k_crv(u)
k_roots = k_crv.roots()
k_infs = np.array(k_crv(k_roots)).transpose()

inflections = inputCrv.GetSamples(k_roots)
crossidx = util.findCrossings(kappa)
crossings = np.array([new_fit_x[crossidx], new_fit_y[crossidx]])


def draw():
    # fitting result
    #plt.plot(input_x, input_y, new_fit_x, new_fit_y)
    #plt.plot(new_fit_x, new_fit_y, res_x, res_y)
    plt.plot(new_fit_x, new_fit_y, crossings[0], crossings[1], '.')

    #plt.plot(input_x, input_y, out[0], out[1], inflections[0], inflections[1], 'o')
    #plt.plot(out[0], out[1])

    plt.axes().set_aspect('equal', 'datalim')

    #plt.figure()
    #plt.plot(u, k_out, k_roots, k_infs, '.')

    plt.show()

draw()