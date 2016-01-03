// -*- mode:c++ coding:utf-8 -*-
/**
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
 */
#include "mylivewire.h"
#include <queue>
#include <vector>
#include <cmath>
#include <fstream>
#include <functional>
#include <assert.h>

#ifdef TEST
#include <iostream>
#endif

struct PriorityQueueElement
{
	double score;
	int elem;
	
	bool operator> (const PriorityQueueElement& rhs) const 
	{ 
		return score > rhs.score;
	}
};

class PriorityQueue 
{
private:
	std::vector<bool> existence_;
	std::priority_queue<PriorityQueueElement, 
						std::vector<PriorityQueueElement>,
						std::greater<PriorityQueueElement> > queue_;
	
public:
	PriorityQueue(int nElem) : existence_(nElem, false)
	{
	}
	
	void add(double score, int elem) {
		PriorityQueueElement e;
		e.score = score;
		e.elem = elem;
		queue_.push(e);
		existence_[elem] = true;
	}
	
	bool empty() const {
		return queue_.empty();
	}
	
	void remove(int elem) {
		existence_[elem] = false;
	}
	
	bool exist(int elem) {
		return existence_[elem];
	}
	
	int pop() {
		int val;
		bool removed;
		while (true) {
			const PriorityQueueElement& e = queue_.top();
			val = e.elem;
			queue_.pop();
			if (existence_[val]) {
				existence_[val] = false;
				return val;
			}
		}
	}
};

void getNeighbors(int& n, int neighbors[8], int idx, int nRow, int nCol)
{
	n = 0;
	std::div_t ret = std::div(idx, nCol);
	int nX = ret.rem;
	int nY = ret.quot;
	
	// (-1, 0)
	if (nX > 1) {
		neighbors[n++] = idx - 1;
	}
	// (1, 0)
	if (nX < nCol - 2) {
		neighbors[n++] = idx + 1;
	}
	// (0, -1)
	if (nY > 1) {
		neighbors[n++] = idx - nCol;
	}
	// (0, 1)
	if (nY < nRow - 2) {
		neighbors[n++] = idx + nCol;
	}
	// (-1, -1)
	if (nX > 1 && nY > 1) {
		neighbors[n++] = idx - nCol - 1;
	}
	// (-1, 1)
	if (nX > 1 && nY < nRow - 2) {
		neighbors[n++] = idx + nCol - 1;
	}
	// (1, -1)
	if (nX < nCol - 2 && nY > 1) {
		neighbors[n++] = idx - nCol + 1;
	}
	// (1, 1)
	if (nX < nCol - 2 && nY < nRow - 2) {
		neighbors[n++] = idx + nCol + 1;
	}
}

double lowcost(int p, int q, int nRow, int nCol, double g, double gmin, double gmax)
{
	std::div_t ret_p = std::div(p, nCol);
	std::div_t ret_q = std::div(q, nCol);
    int dx = ret_p.rem - ret_q.rem;
    int dy = ret_p.quot - ret_q.quot;
    double norm = std::sqrt(dx * dx + dy * dy);
    return (1 - ((g - gmin) / (gmax - gmin)))*norm/1.4142135623730951;
}

void mylivewire(int *path, int row, int col, int s[2], double *gradImg, int row2, int col2)
{
	static const int True = 1;
	static const int False = 0;
	
//#define RECORD
#ifdef RECORD
	{
		std::ofstream ofs("record.txt");
		ofs << s[0] << " " << s[1] << std::endl;
		ofs << row << " " << col << std::endl;
		for (int i=0; i<row * col; ++i) {
			ofs << path[i] << " ";
		}
		ofs << std::endl;
		ofs << row2 << " " << col2 << std::endl;
		for (int i=0; i<row2 * col2; ++i) {
			ofs << gradImg[i] << " ";
		}
		ofs << std::endl;
		ofs.flush();
	}
#endif
	
	int nGradImg = row2 * col2;
	double *g = new double[nGradImg];
	int *e = new int[nGradImg];
	
	memset(g, 0, nGradImg * sizeof(double));
	memset(e, False, nGradImg * sizeof(int));
	
	// find gmin & gmax
	double gmin = std::numeric_limits<double>::max();
	double gmax = std::numeric_limits<double>::min();

	for (int i=0; i<nGradImg; ++i) {
		double t = gradImg[i];
		gmin = std::min(gmin, t);
		gmax = std::max(gmax, t);
	}
	
	int nNeighbors;
	int neighbors[8];
	
	int nS = col * s[1] + s[0];
	path[nS] = nS;
	PriorityQueue pq(nGradImg);
	pq.add(g[nS], nS);
	while (!pq.empty()) {
		int nQ = pq.pop();
		e[nQ] = True;
		getNeighbors(nNeighbors, neighbors, nQ, row2, col2);
		for (int i=0; i<nNeighbors; ++i) {
			int nR = neighbors[i];
			if (!e[nR]) {
				double gtmp = g[nQ] + lowcost(nQ, nR, row2, col2, gradImg[nR], gmin, gmax);
				if (pq.exist(nR)) {
					if (gtmp < g[nR]) {
						pq.remove(nR);
					}
				} else {
					g[nR] = gtmp;
					path[nR] = nQ;
					pq.add(gtmp, nR);
				}
			}
		}
	}	
}

#ifdef TEST
int main (int argc, char const *argv[])
{
	int *path;
	int row;
	int col;
	int s[2];
	double *gradImg;
	int row2;
	int col2;
	std::ifstream ifs("record.txt");
	ifs >> s[0] >> s[1];
	ifs >> row >> col;
	int nSize = row * col;
	path = new int[nSize];
	for (int i=0; i<nSize; ++i) {
		ifs >> path[i];
	}
	ifs >> row2 >> col2;
	assert(row == row2);
	assert(col == col2);
	gradImg = new double[nSize];
	for (int i=0; i<nSize; ++i) {
		ifs >> gradImg[i];
	}

	mylivewire(path, row, col, s, gradImg, row2, col2);
	
	delete[] path;
	delete[] gradImg;
	return 0;
}
#endif
