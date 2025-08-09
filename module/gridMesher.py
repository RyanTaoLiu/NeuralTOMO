import numpy as np
# import scipy.io
import torch

import matplotlib.pyplot as plt
import pyvista as pv

class GridMesh:
    def __init__(self, problem):
        self.mesh = problem.mesh
        self.initMesh()

        if (problem.boundaryCondition != None):
            self.bc = problem.boundaryCondition
            self.initBC()
        if (problem.materialProperty != None):
            self.material = problem.materialProperty
            self.initK()
            self.init_B_C()

    # -----------------------#
    def initMesh(self):
        self.nelx = self.mesh['nelx']
        self.nely = self.mesh['nely']
        self.nelz = self.mesh['nelz']
        self.elemSize = self.mesh['elemSize']
        self.numElems = self.nelx * self.nely * self.nelz
        self.numNodes = (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)
        self.elemNodes = np.zeros((self.numElems, 8))
        self.elemArea = self.elemSize[0] * self.elemSize[1] * self.elemSize[2] * torch.ones((self.numElems))
        self.netArea = torch.sum(self.elemArea)
        nxy = (self.nely + 1) * (self.nelx + 1)

        for elz in range(self.nelz):
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    el = ely + elx * self.nely + elz * self.nelx * self.nely
                    n1 = (self.nely + 1) * elx + ely
                    n2 = (self.nely + 1) * (elx + 1) + ely
                    self.elemNodes[el, :] = np.array(
                        [n1 + 1 + nxy * elz,
                         n2 + 1 + nxy * elz,
                         n2 + nxy * elz,
                         n1 + nxy * elz,
                         n1 + 1 + nxy * (elz + 1),
                         n2 + 1 + nxy * (elz + 1),
                         n2 + nxy * (elz + 1),
                         n1 + nxy * (elz + 1)])
        self.elemNodes = self.elemNodes.astype(int)

        self.nodeXYZ = np.zeros((self.numNodes, 3))
        ctr = 0
        for k in range(self.nelz + 1):
            for i in range(self.nelx + 1):
                for j in range(self.nely + 1):
                    self.nodeXYZ[ctr, 0] = self.elemSize[0] * i
                    self.nodeXYZ[ctr, 1] = self.nely * self.elemSize[1] - self.elemSize[1] * j
                    self.nodeXYZ[ctr, 2] = self.elemSize[2] * k
                    ctr += 1

        self.elemCenters = self.generatePoints()
        self.elemCentersUpSampling = self.generatePoints(2)

        self.bb_xmin, self.bb_xmax = 0, self.nelx * self.elemSize[0]
        self.bb_ymin, self.bb_ymax = 0, self.nely * self.elemSize[1]
        self.bb_zmin, self.bb_zmax = 0, self.nelz * self.elemSize[2]

    # -----------------------#
    def initBC(self):
        self.ndof = self.bc['numDOFPerNode'] * self.numNodes
        self.fixed = self.bc['fixed']

        if 'nonNullElem' in self.bc:
            self.nonNullElem = self.bc['nonNullElem']
            self.nullElem = np.setdiff1d(np.arange(self.numElems), self.nonNullElem)
        else:
            self.nonNullElem = np.arange(self.numElems)
            self.nullElem = None

        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.f = self.bc['force']
        self.numDOFPerElem = 8 * self.bc['numDOFPerNode']
        self.edofMat = np.zeros((self.nelx * self.nely * self.nelz, self.numDOFPerElem), dtype=int)
        dofxy = 3 * (self.nely + 1) * (self.nelx + 1)
        for elz in range(self.nelz):
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    el = ely + elx * self.nely + elz * self.nelx * self.nely
                    n1 = (self.nely + 1) * elx + ely + (self.nely + 1) * (self.nelx + 1) * elz
                    n2 = (self.nely + 1) * (elx + 1) + ely + (self.nely + 1) * (self.nelx + 1) * elz
                    self.edofMat[el, :] = np.array(
                        [3 * n1 + 3, 3 * n1 + 4, 3 * n1 + 5,
                         3 * n2 + 3, 3 * n2 + 4, 3 * n2 + 5,
                         3 * n2, 3 * n2 + 1, 3 * n2 + 2,
                         3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                         3 * n1 + 3 + dofxy, 3 * n1 + 4 + dofxy, 3 * n1 + 5 + dofxy,
                         3 * n2 + 3 + dofxy, 3 * n2 + 4 + dofxy, 3 * n2 + 5 + dofxy,
                         3 * n2 + dofxy, 3 * n2 + 1 + dofxy, 3 * n2 + 2 + dofxy,
                         3 * n1 + dofxy, 3 * n1 + 1 + dofxy, 3 * n1 + 2 + dofxy], dtype=int)

        self.edofMat = self.edofMat.astype(int)

        self.iK = np.kron(self.edofMat, np.ones((self.numDOFPerElem, 1))).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, self.numDOFPerElem))).flatten()
        bK = tuple(np.zeros((len(self.iK))).astype(int))  # batch values
        self.nodeIdx = [bK, self.iK, self.jK]

    # -----------------------#
    def initK(self):
        def getDMatrix(materialProperty):
            E = materialProperty['E']
            nu = materialProperty['nu']
            A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
                          [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]])
            A = A.T
            B = np.array([[1], [nu]])
            k = 1 / 144 * np.dot(A, B)
            k = k.flatten()
            K1 = np.array([[k[0], k[1], k[1], k[2], k[4], k[4]],
                           [k[1], k[0], k[1], k[3], k[5], k[6]],
                           [k[1], k[1], k[0], k[3], k[6], k[5]],
                           [k[2], k[3], k[3], k[0], k[7], k[7]],
                           [k[4], k[5], k[6], k[7], k[0], k[1]],
                           [k[4], k[6], k[5], k[7], k[1], k[0]]])
            K2 = np.array([[k[8], k[7], k[11], k[5], k[3], k[6]],
                           [k[7], k[8], k[11], k[4], k[2], k[4]],
                           [k[9], k[9], k[12], k[6], k[3], k[5]],
                           [k[5], k[4], k[10], k[8], k[1], k[9]],
                           [k[3], k[2], k[4], k[1], k[8], k[11]],
                           [k[10], k[3], k[5], k[11], k[9], k[12]]])
            K3 = np.array([[k[5], k[6], k[3], k[8], k[11], k[7]],
                           [k[6], k[5], k[3], k[9], k[12], k[9]],
                           [k[4], k[4], k[2], k[7], k[11], k[8]],
                           [k[8], k[9], k[1], k[5], k[10], k[4]],
                           [k[11], k[12], k[9], k[10], k[5], k[3]],
                           [k[1], k[11], k[8], k[3], k[4], k[2]]])
            K4 = np.array([[k[13], k[10], k[10], k[12], k[9], k[9]],
                           [k[10], k[13], k[10], k[11], k[8], k[7]],
                           [k[10], k[10], k[13], k[11], k[7], k[8]],
                           [k[12], k[11], k[11], k[13], k[6], k[6]],
                           [k[9], k[8], k[7], k[6], k[13], k[10]],
                           [k[9], k[7], k[8], k[6], k[10], k[13]]])
            K5 = np.array([[k[0], k[1], k[7], k[2], k[4], k[3]],
                           [k[1], k[0], k[7], k[3], k[5], k[10]],
                           [k[7], k[7], k[0], k[4], k[10], k[5]],
                           [k[2], k[3], k[4], k[0], k[7], k[1]],
                           [k[4], k[5], k[10], k[7], k[0], k[7]],
                           [k[3], k[10], k[5], k[1], k[7], k[0]]])
            K6 = np.array([[k[13], k[10], k[6], k[12], k[9], k[11]],
                           [k[10], k[13], k[6], k[11], k[8], k[1]],
                           [k[6], k[6], k[13], k[9], k[1], k[8]],
                           [k[12], k[11], k[9], k[13], k[6], k[10]],
                           [k[9], k[8], k[1], k[6], k[13], k[6]],
                           [k[11], k[1], k[8], k[10], k[6], k[13]]])
            A1 = np.concatenate((K1, K2, K3, K4), axis=1)
            A2 = np.concatenate((K2.T, K5, K6, K3.T), axis=1)
            A3 = np.concatenate((K3.T, K6, K5.T, K2.T), axis=1)
            A4 = np.concatenate((K4, K3, K2, K1.T), axis=1)
            KE = 1 / ((nu + 1) * (1 - 2 * nu)) * np.concatenate((A1, A2, A3, A4), axis=0)
            return (KE)

        self.Emax = self.material['E']
        self.nu = self.material['nu']
        self.penal = self.material['penal']
        self.KE = np.tile(getDMatrix(self.material)[np.newaxis, :, :], (self.numElems, 1, 1))

    def init_B_C(self):
        self.B = np.array([
        [-0.25,     0,     0,  0.25,     0,     0,  0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0, 0.25,    0,    0, -0.25,     0,     0],
        [    0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0,  0.25,     0,     0, -0.25,     0,     0, -0.25,     0,    0, 0.25,    0,     0,  0.25,     0],
        [    0,     0, -0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0,  0.25,    0,    0, 0.25,     0,     0,  0.25],
        [-0.25, -0.25,     0, -0.25,  0.25,     0,  0.25,  0.25,     0,  0.25, -0.25,     0, -0.25, -0.25,     0, -0.25,  0.25,     0, 0.25, 0.25,    0,  0.25, -0.25,     0],
        [    0, -0.25, -0.25,     0, -0.25, -0.25,     0, -0.25,  0.25,     0, -0.25,  0.25,     0,  0.25, -0.25,     0,  0.25, -0.25,    0, 0.25, 0.25,     0,  0.25,  0.25],
        [-0.25,     0, -0.25, -0.25,     0,  0.25, -0.25,     0,  0.25, -0.25,     0, -0.25,  0.25,     0, -0.25,  0.25,     0,  0.25, 0.25,    0, 0.25,  0.25,     0, -0.25]]
        )
        self.G = self.Emax / (2 * (1 + self.nu))

        self.C = np.array(
            [[1/self.Emax,    -self.nu/self.Emax,  -self.nu/self.Emax,  0,      0,      0],
            [-self.nu/self.Emax, 1/self.Emax,      -self.nu/self.Emax,  0,      0,      0],
            [-self.nu/self.Emax, -self.nu/self.Emax,  1/self.Emax,      0,      0,      0],
            [0,         0,          0,          1/self.G,  0,      0],
            [0,         0,          0,          0,      1/self.G,  0],
            [0,         0,          0,          0,      0,      1/self.G]])

    # -----------------------#
    def generatePoints(self, resolution=1):  # generate points in elements
        ctr = 0
        xy = np.zeros((resolution * self.nelx * resolution * self.nely * resolution * self.nelz, 3))

        for k in range(resolution * self.nelz):
            for i in range(resolution * self.nelx):
                for j in range(resolution * self.nely):
                    xy[ctr, 0] = self.elemSize[0] * (i + 0.5) / resolution
                    xy[ctr, 1] = self.elemSize[1] * (resolution * self.nely - j - 0.5) / resolution
                    xy[ctr, 2] = self.elemSize[2] * (k + 0.5) / resolution
                    ctr += 1
        return xy

    # -----------------------#
    def plotField(self, field, titleStr, res=1):
        return
        fig, ax = plt.subplots()
        fig.suptitle(titleStr)
        ax.grid(False)
        ax.axis(False)

        field_reshaped = field.reshape(res * self.nelx, res * self.nelz, res * self.nely).T

        plt.subplot(2, 2, 1)
        plt.axis('Equal')
        plt.grid(False)

        plt.imshow(field_reshaped[res * self.nely // 2, :, :], cmap='gray',
                   interpolation='none', origin='lower')

        plt.subplot(2, 2, 2)
        plt.axis('Equal')
        plt.grid(False)
        plt.imshow(field_reshaped[:, res * self.nelz // 2, :], cmap='gray',
                   interpolation='none', origin='lower')

        plt.subplot(2, 2, 3)
        plt.axis('Equal')
        plt.grid(False)
        plt.imshow(field_reshaped[:, :, res * self.nelx // 2], cmap='gray',
                   interpolation='none', origin='lower')

        fig.canvas.draw()
        plt.pause(0.01)

    def saveField(self, field, normal, savepath, res=1, **kwargs):
        field_reshaped = field.reshape(res * self.nelx, res * self.nelz, res * self.nely, order='F')
        # scipy.io.savemat(savepath[:-4] + '.mat', {'density': field_reshaped})

        fieldPosition = self.elemCenters
        pct = pv.PolyData(fieldPosition, None)
        pct.point_data['density'] = field
        if normal is not None:
            pct.point_data['normal'] = normal.T

        if 'drho_dxyz' in kwargs:
            drho_dxyz = kwargs['drho_dxyz']
            pct.point_data['gradient'] = drho_dxyz.cpu().detach().numpy()

        threshed = pct.threshold(0.1, scalars='density')
        threshed.save(savepath)
