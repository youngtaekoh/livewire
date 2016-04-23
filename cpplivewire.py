# -*- coding: utf-8 -*-
"""
Copyright (c) YoungTaek Oh All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from Tkinter import *
from tkFileDialog import askopenfilename
import mylivewire
import numpy as np
import cv2

#from scipy.interpolate import UnivariateSpline
from scipy import interpolate

def livecontour(t, pathmat):
    row, col = pathmat.shape
    n = pathmat[t]
    prev_n = -1
    ret = []
    while n != prev_n:
        prev_n = n;
        nidx = np.unravel_index(n, pathmat.shape);
        ret.append((nidx[1], nidx[0])) # Reverse for OpenCV
        n = pathmat[nidx]
    ret.pop()
    ret.reverse()
    return np.array(ret)

guiroot = Tk()
guiroot.withdraw()

# Process
imagefilename = askopenfilename(filetypes=[('TIFF files', '*.tif'), ('PNG files', '*.png'), ('JPEG files', '*.jpg')])
if len(imagefilename) == 0:
    sys.exit(-1)
    
img = cv2.imread(imagefilename, 0)
if len(img.shape) > 2:
    bwImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
else:
    bwImg = img
ret, thresh = cv2.threshold(bwImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((7, 7), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
sobelX = cv2.Sobel(thresh, cv2.CV_8U, 1, 0, ksize=3)
sobelY = cv2.Sobel(thresh, cv2.CV_8U, 0, 1, ksize=3)

G = np.sqrt(sobelX.astype('float32')**2 + sobelY.astype('float32')**2)
p = np.zeros(G.shape, dtype=np.int32)

points = []
curves = []
dispImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def computeAndDraw(points, curves, fontscale, pointscale):
#    global unew
#    global fs
#    global out
    tImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for spt in points:
        cv2.circle(tImg, spt, pointscale, (255, 170, 0), -1)
    
    if len(points) > 1:
        cv2.polylines(tImg, curves, False, (255, 0, 0), thickness=2)
        fit_x = np.array([])
        fit_y = np.array([])
        for crv in curves:
            fit_x = np.hstack((fit_x, crv[:, 0]))
            fit_y = np.hstack((fit_y, crv[:, 1]))    
        
        try:                
            tck, u = interpolate.splprep([fit_x, fit_y], k=4)
            
            out = interpolate.splev(u, tck)
            cnt = np.column_stack((out[0], out[1])).astype(np.int32)
            cv2.polylines(tImg, [cnt], False, (0, 0, 255), thickness=1)
            
            unew = np.linspace(0, 1, len(u), endpoint=True)
            dx = interpolate.splev(unew, tck, der=1)
            ddx = interpolate.splev(unew, tck, der=2)
            k_sample = (dx[0]*ddx[1]-ddx[0]*dx[1])/np.linalg.norm(dx)**3
            fs = interpolate.InterpolatedUnivariateSpline(unew, k_sample, k=3)
            roots = fs.roots()
            cv2.putText(tImg, '# of inflection points: %d' % len(roots), (0, int(30 * fontscale)), cv2.FONT_HERSHEY_COMPLEX, fontscale, (255, 255, 255), 1, cv2.CV_AA)
            if (len(roots)>0):
                out = interpolate.splev(roots, tck)
                for ipt in np.transpose(out):
                    cv2.circle(tImg, tuple(ipt.astype(np.int32)), pointscale, (0, 0, 255), -1)

        except ValueError as e:                    
            cv2.putText(tImg, 'Cannot compute: %s' % e, (0, int(30 * fontscale)), cv2.FONT_HERSHEY_COMPLEX, fontscale, (255, 255, 255), 1, cv2.CV_AA)
    return tImg

def click_and_draw(event, x, y, flags, param):
    fontscale = 0.5
    pointscale = 3
    global last_x
    global last_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            if len(points) > 0:
                #cnt = livecontour((y, x), p)
                cnt = livecontour((last_y, last_x), p)
                s = tuple(cnt[-1])
                curves.append(cnt)
            else:
                s = (x, y)   
            
            points.append(s)
            mylivewire.mylivewire(p, s, G)
            tImg = computeAndDraw(points, curves, fontscale, pointscale)
            np.copyto(dispImg, tImg)
            
        else:
            last_x = x
            last_y = y
            if len(points) > 0:
                tImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                for spt in points:
                    cv2.circle(tImg, spt, pointscale, (255, 170, 0), -1)
                if len(curves) > 0:
                    cv2.polylines(tImg, curves, False, (255, 0, 0), thickness=2)

                cnt = livecontour((y, x), p)
                cv2.polylines(tImg, [cnt], False, (255, 0, 0), thickness=2)
                np.copyto(dispImg, tImg)
                
    elif event == cv2.EVENT_RBUTTONDOWN:        
        try:
            points.pop()        
            curves.pop()
        
            s = points[-1]
            mylivewire.mylivewire(p, s, G)
            tImg = computeAndDraw(points, curves, fontscale, pointscale)
            np.copyto(dispImg, tImg)
        except IndexError:
            np.copyto(dispImg, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))            

cv2.namedWindow('livewire')
cv2.setMouseCallback('livewire', click_and_draw)

while True:
    cv2.imshow('livewire', dispImg)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
guiroot.destroy()
cv2.destroyAllWindows()