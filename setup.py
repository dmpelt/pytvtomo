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
import sys
from distutils.version import LooseVersion
from distutils.core import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True
cmdclass = { }
if use_cython:
    ext_modules = [
        Extension("tvtomo.FGPCython",[ "tvtomo/FGPCython.pyx"],extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    ]
    cmdclass = { 'build_ext': build_ext }
else:
    ext_modules = [
        Extension("tvtomo.FGPCython",[ "tvtomo/FGPCython.c" ]),
    ]

setup (name = 'TVTomo',
    version = '1.0',
    description = 'Python library for Total Variation minimization in tomography',
    author='D.M. Pelt',
    author_email='D.M.Pelt@cwi.nl',
    url='http://dmpelt.github.io/tvtomo/',
    #ext_package='astra',
    #ext_modules = cythonize(Extension("astra/*.pyx",extra_compile_args=extra_compile_args,extra_linker_args=extra_compile_args)),
    license='GPLv3',
    ext_modules = ext_modules,
    include_dirs=[np.get_include()],
    cmdclass = cmdclass,
    #ext_modules = [Extension("astra","astra/astra.pyx")],
    packages=['tvtomo'],
    requires=['numpy','six'],
)
