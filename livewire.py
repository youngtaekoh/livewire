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

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import heapq
import math

class PriorityQueueElement(object):
    """ 
    A proxy for an element in a priority queue that remembers (and
    compares according to) its scores.
    """
    def __init__(self, elem, score):
        self._elem = elem
        self._score = score
        self._removed = False
        
    def __lt__(self, other):
        return self._score < other._score
        

class PriorityQueue(object):
    """ 
    A priority queue with O(log n) addition, O(1) membership test and
    amortized O(log n) removal.
    
    The 'add' and 'remove' methods add and remove elements from the
    queue, and the 'pop' method removes and returns the element with
    the lowest score.
    """
    def __init__(self):
        self._heap = []
        self._dict = {}
        
    def __contains__(self, elem):
        return elem in self._dict
        
    def __iter__(self):
        return iter(self._dict)
        
    def empty(self):
        return len(self._dict) == 0
        
    def add(self, score, elem):
        """
        Add an element to a priority queue
        """
        e = PriorityQueueElement(elem, score)
        self._dict[elem] = e
        heapq.heappush(self._heap, e)
        
    def remove(self, elem):
        """
        Remove an element from a priority queue. If the element is not
        a member, raise KeyError.
        """
        e = self._dict.pop(elem)
        e._removed = True
        
    def pop(self):
        """
        Remove and return the element with the smallest score from a
        priority queue. 
        """
        while True:
            e = heapq.heappop(self._heap)
            if not e._removed:
                del self._dict[e._elem]
                return e._elem

def lowcost(p, q, g, gmin, gmax):
    """
    A low cost function
    """
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    norm = math.sqrt(dx ** 2 + dy ** 2)
    return (1 - ((g - gmin) / (gmax - gmin)))*norm/1.4142135623730951
    
def Neighbor(q, gradImg):
    ret = []
    maxX, maxY = gradImg.shape
    for x, y in [ (-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1) ]:
        nx = q[0] + x
        ny = q[1] + y
        if (nx >= 0 and nx < maxX and ny >=0 and ny < maxY):
            ret.append((nx, ny))
    return ret
    
def livewire(s, gradImg):
    L = PriorityQueue()
    g = np.zeros(gradImg.shape, np.float32)
    e = np.zeros(gradImg.shape, np.bool)
    p = np.empty(shape=gradImg.shape, dtype=object)
    gmin = np.min(gradImg)
    gmax = np.max(gradImg)
    L.add(g[s], s)
    while not L.empty():
        q = L.pop()
        e[q] = True
        N = Neighbor(q, gradImg)
        for r in N:
            if not e[r]:
                gtmp = g[q] + lowcost(q, r, gradImg[r], gmin, gmax)
                if r in L:
                    if gtmp < g[r]:
                        L.remove(r)
                else:
                    g[r] = gtmp
                    p[r] = q
                    L.add(gtmp, r)
    return p
    
def livecontour(t, pmat):
    n = pmat[t]
    ret = []
    while n:
        ret.append((n[1], n[0])) # Reverse for OpenCV
        n = pmat[n]
    return np.array(ret)
    
img = cv2.imread('liver.png', 0)
bwImg = img
ret, thresh = cv2.threshold(bwImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((7, 7), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
sobelX = cv2.Sobel(thresh, cv2.CV_8U, 1, 0, ksize=3)
sobelY = cv2.Sobel(thresh, cv2.CV_8U, 0, 1, ksize=3)
G = np.sqrt(sobelX.astype('float32')**2 + sobelY.astype('float32')**2)

s = (80, 141)
p = livewire(s, G)

dispImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def click_and_draw(event, x, y, flags, param):
    global p
    if event == cv2.EVENT_LBUTTONDOWN:
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


                    
        