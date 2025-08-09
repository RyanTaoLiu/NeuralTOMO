from abc import ABC
import numpy as np


class materialBase(ABC):
    def __init__(self):
        self.penal = 3
        self.isotropic = True

    def toDict(self):
        return dict(self)

    def __getitem__(self, item):
        return getattr(self, item)

    def stress2HoffmanF(self):
        tensileStrengthList = np.asarray([self.tensileStrengthX, self.tensileStrengthY, self.tensileStrengthZ])
        compressiveStrengthList = np.asarray(
            [self.compressiveStrengthX, self.compressiveStrengthY, self.compressiveStrengthZ])
        shearStrengthList = np.asarray([self.shearStrengthXY, self.shearStrengthYZ, self.shearStrengthZX])
        F123 = [1 / tensileStrengthList[i] + 1 / compressiveStrengthList[i] for i in range(3)]
        F_11_22_33 = [-1 / (tensileStrengthList[i] * compressiveStrengthList[i]) for i in range(3)]
        # F_12_23_31 = [-0.5 / (shearStrengthList[i] * shearStrengthList[i]) for i in range(3)]
        F_12_23_31 = [-F_11_22_33[0] - F_11_22_33[1] + F_11_22_33[2],
                      F_11_22_33[0] - F_11_22_33[1] - F_11_22_33[2],
                      -F_11_22_33[0] + F_11_22_33[1] - F_11_22_33[2]]
        F_44_55_66 = [1 / (3 * shearStrengthList[i] * shearStrengthList[i]) for i in range(3)]
        return F123 + F_11_22_33 + F_12_23_31

    # from Nonlinear three-dimensional anisotropic material model for failure analysis of timber

    def stress2HoffmanPQ(self):
        tensileStrengthList = np.asarray([self.tensileStrengthX, self.tensileStrengthY, self.tensileStrengthZ])
        compressiveStrengthList = np.asarray(
            [self.compressiveStrengthX, self.compressiveStrengthY, self.compressiveStrengthZ])
        shearStrengthList = np.asarray([self.shearStrengthYZ, self.shearStrengthZX, self.shearStrengthXY])
        F123 = [1 / tensileStrengthList[i] - 1 / compressiveStrengthList[i] for i in range(3)]
        F_11_22_33 = [1 / (2 * tensileStrengthList[i] * compressiveStrengthList[i]) for i in range(3)]
        F_12_23_31 = [F_11_22_33[0] + F_11_22_33[1] - F_11_22_33[2],
                      -F_11_22_33[0] + F_11_22_33[1] + F_11_22_33[2],
                      F_11_22_33[0] - F_11_22_33[1] + F_11_22_33[2]]
        alpha = F_12_23_31
        F_44_55_66 = [1 / (3 * shearStrengthList[i] * shearStrengthList[i]) for i in range(3)]

        P = np.asarray([[2 * (alpha[2] + alpha[0]),   -1 * alpha[0],              -1 * alpha[2],                0, 0, 0],
                      [-1 * alpha[0],               2 * (alpha[1] + alpha[0]),  -1 * alpha[1],                  0, 0, 0],
                      [-1 * alpha[2],               -1 * alpha[1],              2 * (alpha[1] + alpha[2]),      0, 0, 0],
                      [0, 0, 0, 6 * F_44_55_66[0], 0, 0],
                      [0, 0, 0, 0, 6 * F_44_55_66[1], 0],
                      [0, 0, 0, 0, 0, 6 * F_44_55_66[2]]])
        '''
        P = np.asarray([[2 * (alpha[2] + alpha[0]),  0,              0,                0, 0, 0],
              [0,               2 * (alpha[1] + alpha[0]),   0,                0, 0, 0],
              [0,               0,              2 * (alpha[1] + alpha[2]),     0, 0, 0],
              [0, 0, 0, 6 * F_44_55_66[0], 0, 0],
              [0, 0, 0, 0, 6 * F_44_55_66[1], 0],
              [0, 0, 0, 0, 0, 6 * F_44_55_66[2]]])
        '''
        '''
        P = np.asarray([[2 * (alpha[2] + alpha[0]),   -1 * alpha[0],              -1 * alpha[2],                0, 0, 0],
                      [-1 * alpha[0],               2 * (alpha[1] + alpha[0]),  -1 * alpha[1],                  0, 0, 0],
                      [-1 * alpha[2],               -1 * alpha[1],              2 * (alpha[1] + alpha[2]),      0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        '''

        Q = np.asarray(F123 + [0, 0, 0])
        return P, Q


class default():
    def __init__(self):
        super().__init__()
        self.E = 1
        self.nu = 0.3

        self.Ef = self.E
        self.Et = self.E
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = True

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 0.03
        self.compressiveStrengthX = 0.03

        self.tensileStrengthY = 0.03
        self.compressiveStrengthY = 0.03

        self.tensileStrengthZ = 0.03
        self.compressiveStrengthZ = 0.03

        self.shearStrengthXY = 0.03
        self.shearStrengthYZ = 0.03
        self.shearStrengthZX = 0.03

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()


# Anisotropic properties of 3-D printed Poly Lactic Acid (PLA) and
# Acrylonitrile Butadiene Styrene (ABS) plastics

class PLA(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e3
        self.nu = 0.35

        self.Ef = self.E
        self.Et = self.E
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = True

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 52
        self.compressiveStrengthX = 61

        self.tensileStrengthY = 32
        self.compressiveStrengthY = 47

        self.tensileStrengthZ = 18
        self.compressiveStrengthZ = 52

        self.shearStrengthXY = 35
        self.shearStrengthYZ = 35
        self.shearStrengthZX = 35

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()


class PLAPlus(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e3
        self.nu = 0.35

        self.Ef = self.E
        self.Et = self.E
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = True

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 52
        self.compressiveStrengthX = 61

        self.tensileStrengthY = 32
        self.compressiveStrengthY = 47

        self.tensileStrengthZ = 18
        self.compressiveStrengthZ = 52

        self.shearStrengthXY = 24.25
        self.shearStrengthYZ = 14.43
        self.shearStrengthZX = 14.43

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()

# used for elementSize = 2mm
class PLAPlus_Scale2(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e3 / 4
        self.nu = 0.35

        self.Ef = self.E
        self.Et = self.E
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = True

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 52 / 4
        self.compressiveStrengthX = 61 / 4

        self.tensileStrengthY = 32 / 4
        self.compressiveStrengthY = 47 / 4

        self.tensileStrengthZ = 18 / 4
        self.compressiveStrengthZ = 52 / 4

        self.shearStrengthXY = 24.25 / 4
        self.shearStrengthYZ = 14.43 / 4
        self.shearStrengthZX = 14.43 / 4

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()

# used for elementSize = 3mm
class PLAPlus_Scale3(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e3 / 9
        self.nu = 0.35

        self.Ef = self.E
        self.Et = self.E
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = True

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 52 / 9
        self.compressiveStrengthX = 61 / 9

        self.tensileStrengthY = 32 / 9
        self.compressiveStrengthY = 47 / 9

        self.tensileStrengthZ = 18 / 9
        self.compressiveStrengthZ = 52 / 9

        self.shearStrengthXY = 24.25 / 9
        self.shearStrengthYZ = 14.43 / 9
        self.shearStrengthZX = 14.43 / 9

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()

class PLA1(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e9
        self.nu = 0.35

        self.Ef = self.E
        self.Et = self.E
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = True

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 52e6
        self.compressiveStrengthX = 61e6

        self.tensileStrengthY = 32e6
        self.compressiveStrengthY = 47e6

        self.tensileStrengthZ = 18e6
        self.compressiveStrengthZ = 52e6

        self.shearStrengthXY = 35e6
        self.shearStrengthYZ = 35e6
        self.shearStrengthZX = 35e6

        self.F = self.stress2HoffmanF()

class CFPlus(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e3
        self.nu = 0.35

        self.Ef = 9e3
        self.Et = 3e3
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = False

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 152
        self.compressiveStrengthX = 81

        self.tensileStrengthY = 32
        self.compressiveStrengthY = 47

        self.tensileStrengthZ = 18
        self.compressiveStrengthZ = 52

        self.shearStrengthXY = 53.12
        self.shearStrengthYZ = 14.43
        self.shearStrengthZX = 49.08

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()

class CFPlus_Scale2(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e3 / 4
        self.nu = 0.35

        self.Ef = 9e3 / 4
        self.Et = 3e3 / 4
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = False

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 152 / 4
        self.compressiveStrengthX = 81 / 4

        self.tensileStrengthY = 32 / 4
        self.compressiveStrengthY = 47 / 4

        self.tensileStrengthZ = 18 / 4
        self.compressiveStrengthZ = 52 / 4

        self.shearStrengthXY = 53.12 / 4
        self.shearStrengthYZ = 14.43 / 4
        self.shearStrengthZX = 49.08 / 4

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()

class CFPlus_Scale2_5(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e3 / 2.5 / 2.5
        self.nu = 0.35

        self.Ef = 9e3 / 2.5 / 2.5
        self.Et = 3e3 / 2.5 / 2.5
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = False

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 152 / 2.5 / 2.5
        self.compressiveStrengthX = 81 / 2.5 / 2.5

        self.tensileStrengthY = 32 / 2.5 / 2.5
        self.compressiveStrengthY = 47 / 2.5 / 2.5

        self.tensileStrengthZ = 18 / 2.5 / 2.5
        self.compressiveStrengthZ = 52 / 2.5 / 2.5

        self.shearStrengthXY = 53.12 / 2.5 / 2.5
        self.shearStrengthYZ = 14.43 / 2.5 / 2.5
        self.shearStrengthZX = 49.08 / 2.5 / 2.5

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()

class CFPlus_Scale3(materialBase):
    def __init__(self):
        super().__init__()
        self.E = 3e3 / 9
        self.nu = 0.35

        self.Ef = 9e3 / 9
        self.Et = 3e3 / 9
        self.nuf = self.nu
        self.nut = self.nu

        self.penal = 3
        self.isotropic = False

        # 1e6 <=> 1Mpa
        self.tensileStrengthX = 152 / 9
        self.compressiveStrengthX = 81 / 9

        self.tensileStrengthY = 32 / 9
        self.compressiveStrengthY = 47 / 9

        self.tensileStrengthZ = 18 / 9
        self.compressiveStrengthZ = 52 / 9

        self.shearStrengthXY = 53.12 / 9
        self.shearStrengthYZ = 14.43 / 9
        self.shearStrengthZX = 49.08 / 9

        self.F = self.stress2HoffmanF()
        self.P, self.Q = self.stress2HoffmanPQ()

if __name__ == '__main__':
    pla = PLAPlus()
    print(pla.F)
