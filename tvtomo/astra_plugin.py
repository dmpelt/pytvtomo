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
import astra
from .FISTA import FISTA
import numpy as np

class plugin(astra.plugin.ReconstructionAlgorithm2D):
    """Reconstructs with TV-minimization (l1 minimization of the gradient),
    using FISTA [1].
    
    Options:
    
    'tv_reg': strength of l1-term in objective function
    'fgp_iter' (optional): number of iterations to use in each FGP step
    'bmin' (optional): minimum value constraint on pixels
    'bmax' (optional): maximum value constraint on pixels
    'print_progress' (optional): whether to print progress during computation
    'fgp_nthreads' (optional): number of threads to use during FBP step
    
    [1] Beck, A., & Teboulle, M. (2009). Fast gradient-based algorithms for constrained 
        total variation image denoising and deblurring problems. Image Processing, IEEE 
        Transactions on, 18(11), 2419-2434.
    """

    astra_name = "TV-FISTA"
    
    def initialize(self, cfg, tv_reg, fgp_iters=100, bmin=-np.inf, bmax=np.inf, print_progress=False, fgp_nthreads=None):
        self.w = astra.OpTomo(self.pid)
        self.tv = tv_reg
        self.fgp_iters = fgp_iters
        self.pr = print_progress
        self.bmin = bmin
        self.bmax = bmax
        self.nthreads = fgp_nthreads

    def run(self, iterations):
        f = FISTA(self.w,self.tv,self.fgp_iters, bmin=self.bmin, bmax=self.bmax, nthreads=self.nthreads)
        rc = f.reconstruct(self.s, iterations, progress=self.pr)
        self.v[:] = rc