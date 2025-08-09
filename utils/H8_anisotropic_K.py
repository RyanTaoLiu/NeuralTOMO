"""
# =============================================================================
# Write the stiffness matrix of finite element to file. The created file name
# is equal to the string between the underscores of *this* file's name, plus a
# 'K' extension, e.g.,
#
#     python ELEM_K.py
#
# gives a file named ELEM.K in the same directory.
#
# Author: William Hunter
# Copyright (C) 2008, 2015, William Hunter.
# =============================================================================
"""
from __future__ import division

import os

from sympy import symbols, Matrix, diff, integrate, zeros, cos, sin
from numpy import abs, array

# from ..utils import get_logger, get_data_file
# from .matlcons import *

_a, _b, _c = 0.5, 0.5, 0.5  # element dimensions (half-lengths) don't change!
_E  = 1  # modulus of elasticity
_nu = 1 / 3  # poisson's ratio
_G = _E / (2 * (1 + _nu))  # modulus of rigidity
_g = _E / ((1 + _nu) * (1 - 2 * _nu))
_k = 1  # thermal conductivity of steel = 50 (ref. Mills)


# for Anisotropic
_E_f = 5
_E_t = 1
_nu_f = 0.3
_nu_t = 0.32
_G_f = _E_f / (2 * (1 + _nu_f))
_G_t = _E_t / (2 * (1 + _nu_t))
_phi = 0.78539816339
_theta = 0.78539816339 # pi/4

# logger = get_logger(__name__)
# Get file name:
# fname = get_data_file(__file__)

#if os.path.exists(fname):
    #logger.info('{} (stiffness matrix) exists!'.format(fname))
#else:
    # SymPy symbols:
#a, b, c, x, y, z = symbols('a b c x y z')

#a, b, c = 0.5, 0.5, 0.5
a, b, c = 0.5, 0.5, 0.5
x, y, z = symbols('x y z')

N1, N2, N3, N4 = symbols('N1 N2 N3 N4')
N5, N6, N7, N8 = symbols('N5 N6 N7 N8')
#E, nu, g, G = symbols('E nu g G')
E_f, nu_f, G_f = symbols('Ef nuf Gf')
E_t, nu_t, G_t = symbols('Et nut Gt')
theta, phi = symbols('theta phi')


o = symbols('o') #  dummy symbol
xlist = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
ylist = [y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y]
zlist = [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z]
yxlist = [y, x, o, y, x, o, y, x, o, y, x, o, y, x, o, y, x, o, y, x, o, y, x, o]
zylist = [o, z, y, o, z, y, o, z, y, o, z, y, o, z, y, o, z, y, o, z, y, o, z, y]
zxlist = [z, o, x, z, o, x, z, o, x, z, o, x, z, o, x, z, o, x, z, o, x, z, o, x]

# Shape functions:
N1 = (a - x) * (b - y) * (c - z) / (8 * a * b * c)
N2 = (a + x) * (b - y) * (c - z) / (8 * a * b * c)
N3 = (a + x) * (b + y) * (c - z) / (8 * a * b * c)
N4 = (a - x) * (b + y) * (c - z) / (8 * a * b * c)
N5 = (a - x) * (b - y) * (c + z) / (8 * a * b * c)
N6 = (a + x) * (b - y) * (c + z) / (8 * a * b * c)
N7 = (a + x) * (b + y) * (c + z) / (8 * a * b * c)
N8 = (a - x) * (b + y) * (c + z) / (8 * a * b * c)

# Create strain-displacement matrix B:
B0 = tuple(map(diff, [N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0,\
                N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0, 0], xlist))
B1 = tuple(map(diff, [0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0,\
                0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0], ylist))
B2 = tuple(map(diff, [0, 0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4,\
                0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8], zlist))
B3 = tuple(map(diff, [N1, N1, N1, N2, N2, N2, N3, N3, N3, N4, N4, N4,\
                N5, N5, N5, N6, N6, N6, N7, N7, N7, N8, N8, N8], yxlist))
#-----old version-----
B4 = tuple(map(diff, [N1, N1, N1, N2, N2, N2, N3, N3, N3, N4, N4, N4,\
                N5, N5, N5, N6, N6, N6, N7, N7, N7, N8, N8, N8], zylist))
B5 = tuple(map(diff, [N1, N1, N1, N2, N2, N2, N3, N3, N3, N4, N4, N4,\
                N5, N5, N5, N6, N6, N6, N7, N7, N7, N8, N8, N8], zxlist))
'''
# Jiangyu version
B4 = tuple(map(diff, [N1, N1, N1, N2, N2, N2, N3, N3, N3, N4, N4, N4,\
                N5, N5, N5, N6, N6, N6, N7, N7, N7, N8, N8, N8], zxlist))
B5 = tuple(map(diff, [N1, N1, N1, N2, N2, N2, N3, N3, N3, N4, N4, N4,\
                N5, N5, N5, N6, N6, N6, N7, N7, N7, N8, N8, N8], zylist))
'''

B = Matrix([B0, B1, B2, B3, B4, B5])

# Create constitutive (material property) matrix:
C0 = Matrix([[1/E_f,    -nu_f/E_f,  -nu_f/E_f,  0,      0,      0],
            [-nu_f/E_f, 1/E_t,      -nu_t/E_t,  0,      0,      0],
            [-nu_f/E_f, -nu_t/E_t,  1/E_t,      0,      0,      0],
            [0,         0,          0,          1/G_t,  0,      0],
            [0,         0,          0,          0,      1/G_f,  0],
            [0,         0,          0,          0,      0,      1/G_f]])

C0 = C0.subs({E_f:_E_f, nu_f:_nu_f, G_f:_G_f,
        E_t:_E_t, nu_t:_nu_t, G_t:_G_t})

#cosT, sinT = cos(theta), sin(theta)
#cosP, sinP = cos(phi), sin(phi)
cosT, sinT, cosP, sinP = symbols('cosT sinT cosP sinP')

T1 = Matrix([[cosT*cosT,    sinT*sinT,  0,      0,      0,      -2*cosT*sinT],
             [sinT*sinT,    cosT*cosT,  0,      0,      0,      2*cosT*sinT],
             [0,            0,          1,      0,      0,      0],
             [0,            0,          0,      cosT,   sinT,   0],
             [0,            0,          0,      -sinT,  cosT,   0],
             [sinT*cosT,    -sinT*cosT, 0,      0,      0,      cosT*cosT-sinT*sinT]])

T2 = Matrix([[cosP*cosP, 0,  sinP*sinP, 0,      -2*sinP*cosP,   0],
            [0,         1,  0,          0,      0,              0],
            [sinP*sinP, 0,  cosP*cosP,  0,      2*sinP*cosP,    0],
            [0,         0,  0,          cosP,   0,              sinP],
            [sinP*cosP, 0,  -sinP*cosP, 0,      cosP*cosP-sinP*sinP,0],
            [0,         0,  0,          -sinP,  0,              cosP]])
'''
T1 = Matrix([[cosT**2, sinT**2, 0, -2 * cosT * sinT, 0, 0],
            [sinT**2, cosT**2, 0, 2 * cosT * sinT, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [cosT * sinT, -cosT * sinT, 0, cosT**2 - sinT**2, 0, 0],
            [0, 0, 0, 0, cosT, -sinT],
            [0, 0, 0, 0, sinT, cosT]])

T2 = Matrix([[1, 0, 0, 0, 0, 0],
            [0, cosP**2, sinP**2, 0, 0, -2 * cosP * sinP],
            [0, sinP**2, cosP**2, 0, 0, 2 * cosP * sinP],
            [0, 0, 0, cosP, -sinP, 0],
            [0, 0, 0, sinP, cosP, 0],
            [0, cosP * sinP, -cosP * sinP, 0, 0, cosP**2 - sinP**2]])
'''
C = T2 * T1 * C0.inv() * T1.T * T2.T
dK = B.T * C * B

# Integration:
# logger.info('SymPy is integrating: K for H8...')
# K = dK.integrate((x, -a, a),(y, -b, b),(z, -c, c))
# K = dK.integrate((x, -0.5, 0.5),(y, -0.5, 0.5),(z, -0.5, 0.5))

_K = zeros(24,24)
with open('out1.txt', 'w') as f:
    for i in range(24):
        for j in range(24):
            _K[i,j] = dK[i,j].integrate((x, -0.5, 0.5), (y, -0.5, 0.5), (z, -0.5, 0.5)).simplify()
            #print("[{},{}]:{}".format(i, j, str(_K[i,j])))
            f.write("[{},{}]:{}\n".format(i, j, str(_K[i,j])))
            l_mul_1728 = (_K[i,j]*1728).simplify()
            print("{},".format(l_mul_1728))

K = _K.subs({phi:_phi, theta:_theta})
# Convert SymPy Matrix to NumPy array:
# K = array(_K.subs({phi: _phi, theta: _theta})).astype('double')

# Set small (<< 0) values equal to zero:
#K[abs(K) < 1e-6] = 0

# Create file:
#K.dump(fname)
#logger.info('Created ' + fname + ' (stiffness matrix).')

# EOF H8_K.py
