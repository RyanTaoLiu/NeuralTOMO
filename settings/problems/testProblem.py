import os.path

from settings.problems.problemBase import problemBase

import numpy as np


class testProblem(problemBase):
    problemName = 'testProblem'

    def __init__(self):
        super().__init__()
        self.name = 'testProblem'
        self.mesh, self.boundaryCondition, self.materialProperty = self.mbbSettings()


    def mbbSettings(self):
        ### Mesh
        nelx = 55  # number of FE elements along X
        nely = 50  # number of FE elements along Y
        nelz = 60  # number of FE elements along Z
        elemSize = np.array([1.0, 1.0, 1.0])
        mesh = {'nelx': nelx,
                'nely': nely,
                'nelz': nelz,
                'elemSize': elemSize,
                'type': 'grid'}

        ### Material
        matProp = {'E': 1.0,
                   'nu': 0.3,
                   'Ef': 5.0,
                   'Et': 1.0,
                   'nuf': 0.3,
                   'nut': 0.3,
                   'penal': 3}  # Structural

        exampleName = 'testProblem'
        physics = 'Structural'

        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        dofs = np.arange(ndof)
        [il, jl, kl] = np.meshgrid(nelx, 0, np.arange(0, nelz + 1))
        # form matlab index to python index
        force_nid = kl * (nelx + 1) * (nely + 1) + il * (nely + 1) + (nely + 1 - jl) - 1
        force_dof = 3 * force_nid.flatten() + 1
        force = np.zeros((ndof, 1))
        force[force_dof, 0] = 1

        # set fix node id
        [iif, jf, kf] = np.meshgrid(0, np.arange(0, nely + 1), np.arange(0, nelz + 1))
        fixed_nid = (kf * (nelx + 1) * (nely + 1) + iif * (nely + 1) + (
                    nely + 1 - jf)).flatten() - 1  # form matlab index to python index
        fixed = np.concatenate((3 * fixed_nid, 3 * fixed_nid + 1, 3 * fixed_nid + 2))

        bc = {'exampleName': exampleName, 'physics': physics,
              'force': force, 'fixed': fixed, 'numDOFPerNode': 3}

        return mesh, bc, matProp

if __name__ == '__main__':
    mbb_problem = testProblem()
    # savePath = os.path.join('data', 'settings', '{}.npy'.format(mbb_problem.name))
    # mbb_problem.serialize(savePath)

    from module.gridMesher import GridMesh
    mesh = GridMesh(mbb_problem)