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

import mylivewire
import numpy as np
import cv2

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
    return np.array(ret)

img = cv2.imread('../liver.png', 0)
bwImg = img
ret, thresh = cv2.threshold(bwImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((7, 7), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
sobelX = cv2.Sobel(thresh, cv2.CV_8U, 1, 0, ksize=3)
sobelY = cv2.Sobel(thresh, cv2.CV_8U, 0, 1, ksize=3)

G = np.sqrt(sobelX.astype('float32')**2 + sobelY.astype('float32')**2)
p = np.zeros(G.shape, dtype=np.int32)

s = None
dispImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def click_and_draw(event, x, y, flags, param):
    global p

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            global s
            s = (x, y)
            mylivewire.mylivewire(p, s, G)
            
        else:
            if s != None:
                cnt = livecontour((y, x), p)
                tImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.polylines(tImg, [cnt], False, (255, 0, 0), thickness=3)
                np.copyto(dispImg, tImg)

cv2.namedWindow('livewire')
cv2.setMouseCallback('livewire', click_and_draw)

while True:
    cv2.imshow('livewire', dispImg)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()