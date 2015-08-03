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
import tvtomo
import numpy as np
import astra

# Simple phantom
x = np.zeros((1024,1024))
x[400:600,400:600]=1

# ASTRA geometries, projector, and OpTomo
proj_geom = astra.create_proj_geom('parallel',1.0,1024,np.linspace(0,np.pi,180))
vol_geom = astra.create_vol_geom(1024)
pid = astra.create_projector('cuda',proj_geom,vol_geom)
p = astra.OpTomo(pid)

# Create sinogram and add noise
sino = (p*x).reshape(p.sshape)
sino_with_noise = astra.add_noise_to_sino(sino,10**2)

# Define parameters and FISTA object that performs reconstruction
lam = 2**-5
f = tvtomo.FISTA(p,lam,100)

# Actually reconstruct (takes time)
tvrec = f.reconstruct(sino_with_noise,50, progress=True)

# ASTRA data for reconstructions below
sid = astra.data2d.create('-sino', proj_geom,sino_with_noise)
rid = astra.data2d.create('-vol', vol_geom)

#ASTRA SIRT
cfg = astra.astra_dict('SIRT_CUDA')
cfg['ProjectionDataId'] = sid
cfg['ReconstructionDataId'] = rid
sirt_alg = astra.algorithm.create(cfg)
astra.algorithm.run(sirt_alg,100)
sirtrec = astra.data2d.get(rid)


#ASTRA FBP
cfg = astra.astra_dict('FBP_CUDA')
cfg['ProjectionDataId'] = sid
cfg['ReconstructionDataId'] = rid
fbp_alg = astra.algorithm.create(cfg)
astra.algorithm.run(fbp_alg)
fbprec = astra.data2d.get(rid)

import pylab as pl
pl.gray()
import functools
imshow_with_limits = functools.partial(pl.imshow,vmin=0,vmax=1)

pl.subplot(221)
imshow_with_limits(x)
pl.title('Phantom')
pl.axis('off')
pl.subplot(222)
imshow_with_limits(tvrec)
pl.title('TV-MIN, lambda='+str(lam))
pl.axis('off')
pl.subplot(223)
imshow_with_limits(sirtrec)
pl.title('SIRT')
pl.axis('off')
pl.subplot(224)
imshow_with_limits(fbprec)
pl.title('FBP')
pl.axis('off')
pl.show()
