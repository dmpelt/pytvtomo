#-----------------------------------------------------------------------
#Copyright 2015 Daniel M. Pelt
#
#Contact: D.M.Pelt@cwi.nl
#Website: http://www.dmpelt.com
#
#
#This file is part of the PyTV-TOMO, a Python library for Total Variation
#minimization in tomography.
#
#PyTV-TOMO is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#PyTV-TOMO is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with PyTV-TOMO. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------
import numpy as np
cimport numpy as np
from libc.math cimport sqrt,fabs
cimport cython
cimport cython.parallel
cimport openmp
from six.moves import range as prange

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef tvfullloop(float [:,::1] b, float [:,::1] r, float [:,::1] s, float [:,::1] p, float [:,::1] q, float [:,::1] res, float bmin,float bmax, float lam, int m, int n, float t, float tn):
    cdef int i,j
    cdef float lrs, pn, qn, sqr
    for i in cython.parallel.prange(m,nogil=True):
        for j in range(n):
            lrs=0
            if i != m-1: lrs = lrs+r[i,j]
            if j != n-1: lrs = lrs+s[i,j]
            if i != 0: lrs = lrs-r[i-1,j]
            if j != 0: lrs = lrs-s[i,j-1]
            res[i,j] = b[i,j] - lam*lrs
            if res[i,j]<bmin: res[i,j]=bmin
            if res[i,j]>bmax: res[i,j]=bmax
    for i in cython.parallel.prange(m,nogil=True):
        for j in range(n):
            if i!=m-1: pn = r[i,j] + (res[i,j] - res[i+1,j])/(8*lam);
            if j!=n-1: qn = s[i,j] + (res[i,j] - res[i,j+1])/(8*lam);
            if i!=m-1 and j!=n-1:
                sqr = sqrt(pn*pn+qn*qn)
                if sqr>1:
                    pn=pn/sqr
                    qn=qn/sqr
            if i==m-1:
                sqr = fabs(qn)
                if sqr>1: qn = qn/sqr
            if j==n-1:
                sqr = fabs(pn)
                if sqr>1: pn = pn/sqr
            if i!=m-1:
                r[i,j] = pn + (t-1)*(pn-p[i,j])/tn
                p[i,j] = pn
            if j!=n-1:
                s[i,j] = qn + (t-1)*(qn-q[i,j])/tn
                q[i,j] = qn


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef tvendloop(float [:,::1] b, float [:,::1] p, float [:,::1] q, float [:,::1] res, float bmin,float bmax, float lam, int m, int n):
    cdef int i,j
    cdef float lrs
    for i in cython.parallel.prange(m,nogil=True):
        for j in range(n):
            lrs=0
            if i != m-1: lrs = lrs+p[i,j]
            if j != n-1: lrs = lrs+q[i,j]
            if i != 0: lrs = lrs-p[i-1,j]
            if j != 0: lrs = lrs-q[i,j-1]
            res[i,j] = b[i,j] - lam*lrs
            if res[i,j]<bmin: res[i,j]=bmin
            if res[i,j]>bmax: res[i,j]=bmax

class FGPCython(object):

    def __init__(self, lam, fgpiters, shape, bmin=-np.inf, bmax=np.inf, nthreads=None):
        self.lam = lam
        self.nit = fgpiters
        self.bmin = bmin
        self.bmax = bmax
        m = shape[0]
        n = shape[1]
        self.r = np.zeros((m-1,n),dtype=np.float32)
        self.s = np.zeros((m,n-1),dtype=np.float32)
        self.p = np.zeros_like(self.r)
        self.q = np.zeros_like(self.s)
        self.lrs = np.zeros(shape,dtype=np.float32)
        self.res = np.zeros(shape,dtype=np.float32)
        self.nthreads = nthreads

    def initialize(self):
        self.t=1.

    def apply(self, x, p, s, xo):
        xn = self.fgp(x)
        tnew = (1.+np.sqrt(1+4*self.t))/2.
        y = xn + (self.t-1)*(xn-xo)/tnew
        self.t = tnew
        return xn, y

    def fgp(self, b):
        if self.nthreads:
            ot = openmp.omp_get_num_threads()
            openmp.omp_set_num_threads(self.nthreads)
        m, n = b.shape
        lam = self.lam
        bmin = self.bmin
        bmax = self.bmax
        b = b.astype(np.float32)
        self.r.fill(0)
        self.s.fill(0)
        self.p.fill(0)
        self.q.fill(0)
        self.lrs.fill(0)
        self.res.fill(0)
        t = 1.
        for _ in prange(self.nit):
            tn = (1.+np.sqrt(1+4*t*t))/2.
            tvfullloop(b, self.r, self.s, self.p, self.q, self.res, bmin, bmax, lam, m, n, t, tn)
            t = tn
        tvendloop(b,self.p,self.q,self.res,bmin,bmax,lam,m,n)
        if self.nthreads:
            openmp.omp_set_num_threads(ot)
        return self.res.copy()
