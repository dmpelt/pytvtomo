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
from .FISTA import FISTA
try:
    from .astra_plugin import plugin
except ImportError:
    pass