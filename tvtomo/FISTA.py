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
from .FGPCython import FGPCython
from .Utils import fullcost
import numpy as np
from six.moves import range
import six
import time

class FISTA(object):
    """
    Implementation of the (M)FISTA method for tomography, based on [1].

    [1] Beck, A., & Teboulle, M. (2009). Fast gradient-based algorithms for
        constrained total variation image denoising and deblurring problems.
        Image Processing, IEEE Transactions on, 18(11), 2419-2434.
    """

    def __init__(self,p, lam, FGPIters, bmin=-np.inf, bmax=np.inf, FGPClass=FGPCython, monotone=True, nthreads=None):
        self.fgp = FGPClass(lam, FGPIters, p.vshape, bmin=bmin, bmax=bmax, nthreads=nthreads)
        self.p = p
        self.monotone = monotone
        self.lam = lam

    def reconstruct(self,sinogram,iterations,x0=None, progress=False):
        p = self.p
        y = np.zeros(p.vshape)
        if x0 is None:
            rec = np.zeros_like(y)
        else:
            rec = x0.copy()
        if self.monotone==True:
            f = fullcost(rec,p,sinogram,self.lam)
        div = sinogram.size
        t=1.
        starttime = time.time()
        for i in range(iterations):
            target = y + (p.T*(sinogram.flatten() - p*y)).reshape(p.vshape)/div
            z = self.fgp.fgp(target)
            if self.monotone==False:
                xn = z
            else:
                fn = fullcost(z,p,sinogram,self.lam)
                if fn<f:
                    xn = z
                    f = fn
                else:
                    xn = rec
            tn = (1.+np.sqrt(1+4*t))/2.
            y = xn + t*(z-xn)/tn + (t-1)*(xn-rec)/tn
            t = tn
            rec = xn
            if progress:
                timeleft = int((iterations-i-1)*(time.time()-starttime)/(i+1))
                hours, rem = divmod(timeleft,3600)
                minutes, secs = divmod(rem, 60)
                six.print_("{:.2f} % done ({}:{:02d}:{:02d})".format(100*float(i)/iterations,hours,minutes,secs))
        return rec
