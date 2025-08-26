import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# from sksparse.cholmod import cholesky

import torch
from module.gridMesher import GridMesh

from module.Km2Compliance import *

#from anisotropicFE import angle2Ke
from module.anisotropicFE_new import H8_anisotropic_K
# from quadMesher import QuadMesh
# -----------------------#

class FE:
    # -----------------------#
    def __init__(self, problem, device='cuda'):
        if problem.mesh['type'] == 'grid':
            self.mesh = GridMesh(problem)
        # elif(mesh['type'] == 'quad'):
        #    self.mesh = QuadMesh(mesh, matProp, bc)
        self.init_Matrix_idx(device)

        if type(problem.materialProperty) is dict:
            self.H8 = H8_anisotropic_K(device, **problem.materialProperty)
        else:
            self.H8 = H8_anisotropic_K(device, **problem.materialProperty.__dict__)

    def init_Matrix_idx(self, device):
        self.Ksize = self.mesh.ndof - self.mesh.fixed.flatten().shape[0]
        row_indices_np = self.mesh.iK  # NumPy array of row indices
        col_indices_np = self.mesh.jK  # NumPy array of column indices'

        # keep_index = np.delete(np.arange(0, self.mesh.ndof, dtype=int), self.mesh.fixed)
        keep_index = self.mesh.free
        mask = ~(np.isin(row_indices_np, self.mesh.fixed) | np.isin(col_indices_np, self.mesh.fixed))

        filtered_row_indices = row_indices_np[mask].astype(int)
        filtered_col_indices = col_indices_np[mask].astype(int)
        self.valid_mask = torch.from_numpy(mask).bool().to(device)

        ## ** Need to make sure Array not out of bounds ** ##
        indexMap = np.zeros(self.mesh.ndof, dtype=int)
        indexMap[keep_index] = np.arange(0, self.Ksize, dtype=int)

        # Map filtered indices to new indices in keep_index
        self.new_row_indices = indexMap[filtered_row_indices]
        self.new_col_indices = indexMap[filtered_col_indices]

        self.f = torch.tensor(self.mesh.f[keep_index, 0], dtype=torch.float32, device=device)


    def solve_c_new(self, phi, theta, density, penal=3, isotropic=False):
        # self.u = torch.zeros((self.mesh.ndof, 1), device=density.device)
        if isotropic:
            ## isotropic
            E = self.mesh.Emax * (1e-3 + density) ** self.mesh.penal
            KE = torch.tensor(self.mesh.KE, dtype=torch.float32, device=density.device)
            sK = torch.einsum('i,ijk->ijk', E, KE).flatten()
        else:
            ## anisotropic
            sK = self.H8.angle2Ke(phi, theta, density, penal).flatten()

        d = sK[self.valid_mask]

        f = self.mesh.f[self.mesh.free, 0]
        c = sk2c(d, (self.new_row_indices, self.new_col_indices), f, self.f, self.Ksize)
        return c

    def solve_stress_new(self, phi, theta, density, penal=3, isotropic=False):
        self.u = torch.zeros((self.mesh.ndof, 1), dtype=torch.float32, device=density.device)
        if isotropic:
            ## isotropic
            E = self.mesh.Emax * ((1e-3 + density)*10) ** self.mesh.penal
            KE = torch.tensor(self.mesh.KE, dtype=torch.float32, device=density.device)
            sK = torch.einsum('i,ijk->ijk', E, KE).flatten()
            B = torch.tensor(self.mesh.B.T, dtype=torch.float32, device=density.device).T
            C = torch.tensor(self.mesh.C, dtype=torch.float32, device=density.device).expand(self.mesh.numElems,-1,-1)
        else:
            ## anisotropic
            sK = self.H8.angle2Ke(phi, theta, density, penal).flatten()

            B = self.H8.NodeB
            C = self.H8.temp_C
            T = self.H8.T

        #i = self.sparseKIdx
        d = sK[self.valid_mask]

        f = self.mesh.f[self.mesh.free, 0]
        f_times = 1000
        u = sk2u(d, (self.new_row_indices, self.new_col_indices), f * f_times, self.f * f_times, self.Ksize)
        self.u[self.mesh.free, 0] = u
        c = (self.f*u).sum()
        uElem = self.u[self.mesh.edofMat].reshape(self.mesh.numElems, self.mesh.numDOFPerElem)
        sigmaElem = torch.einsum('bij,jk,bk -> bi', C, B, uElem)
        
        # sigmaElem = torch.einsum('bij,jk,bk,bim -> bm', C, B, uElem, T)

        P, Q = self.H8.P, self.H8.Q
        _A = 0.5 * torch.einsum('ij, jk, ik ->i', sigmaElem, P, sigmaElem)
        _B = torch.einsum('i, ji ->j', Q, sigmaElem)
        _C = -1

        root = torch.zeros_like(_A)
        is_linear = _A < 1e-5

        # the max Force that element can retain
        BL = _B[is_linear]
        BNL = _B[~is_linear]
        ANL = _A[~is_linear]
        root[is_linear] = 1 / (BL.abs() + 1e-5)
        root[~is_linear] = (-BNL + (BNL * BNL - 4 * ANL * _C).sqrt()) / (2 * ANL)


        Fmin = torch.linalg.vector_norm(root.abs() + 1e-5, ord=-6) # + 1e-5*root.mean()
        # Fmin = torch.linalg.vector_norm(root.abs(), ord=-10)
        FRealMin = root.min()
        Criterion = _A * FRealMin * FRealMin + _B * FRealMin + _C

        self.FRealMin = FRealMin
        self.Criterion = Criterion

        if torch.isnan(Fmin):
            print('Fmin is NAN!')
        return Fmin, c

    def solve(self, density):
        self.u=np.zeros((self.mesh.ndof,1))
        E = self.mesh.material['E']*(1.0e-3+density)**self.mesh.material['penal']
        sK = np.einsum('i,ijk->ijk',E, self.mesh.KE).flatten()

        K = coo_matrix((sK,(self.mesh.iK,self.mesh.jK)),shape=(self.mesh.ndof,self.mesh.ndof)).tocsc()
        K = self.deleterowcol(K,self.mesh.fixed,self.mesh.fixed).tocsc()

        B = self.mesh.f[self.mesh.free,0]
        B = scipy.sparse.linalg.spsolve(K, B)
        self.u[self.mesh.free,0]=np.array(B)
        uElem = self.u[self.mesh.edofMat].reshape(self.mesh.numElems,self.mesh.numDOFPerElem)
        self.Jelem  = np.einsum('ik,ik->i',np.einsum('ij,ijk->ik',uElem, self.mesh.KE),uElem)
        return self.u, self.Jelem