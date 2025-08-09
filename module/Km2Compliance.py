import numpy as np
from scipy.sparse import csc_matrix, eye

from sksparse.cholmod import cholesky
# import scipy.sparse

import torch

import platform
import subprocess



import mkl
import pyMKL
mkl.set_num_threads(16)
useMKL = True

class Sk2Complicance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sK, indices, f, f_th, Ksize):
        '''
        :param ctx: Torch context
        :param sK: Torch array
        :param indices: numpy array of index
        :param f: Torch array of Force
        :param f_th: numpy array of Force(i.e. f.detach().cpu().numpy())
        :param Ksize: size of matrix
        :return: compliance c
        '''
        # K = torch.sparse_coo_tensor(indices=sparseIdx, values=sK, size=(Ksize, Ksize))
        row_indices, col_indices = indices
        ## use sksparse
        K_cpu = csc_matrix((sK.detach().cpu().numpy(),
                            (row_indices, col_indices)),
                           shape=(Ksize, Ksize))
        # K_cpu += eye(Ksize)*1e-7 ## do not add this!

        try:
            factor = cholesky(K_cpu)
            B_cpu = factor(f)

        except Exception as e:
            K_cpu += eye(Ksize) * 1e-7
            factor = cholesky(K_cpu)
            B_cpu = factor(f)

        U = torch.tensor(B_cpu, dtype=torch.float32, device=sK.device, requires_grad=False)
        _manual_grad_c = -U[row_indices] * U[col_indices]
        # grad_manual_u = 2*f_th_e
        ctx.save_for_backward(_manual_grad_c)

        c = torch.dot(U, f_th)
        return c

    @staticmethod
    def backward(ctx, grad_c):
        _manual_grad_c, = ctx.saved_tensors
        return _manual_grad_c * grad_c, None, None, None, None


def sk2c(sK: torch.tensor, indices, f, f_th, Ksize):
    return Sk2Complicance.apply(sK, indices, f, f_th, Ksize)


class Sk2Displacement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sK, indices, f, f_th, Ksize):
        '''
        :param ctx: Torch context
        :param sK: Torch array
        :param indices: numpy array of index
        :param f: Torch array of Force
        :param f_th: numpy array of Force(i.e. f.detach().cpu().numpy())
        :param Ksize: size of matrix
        :return: displacement U
        '''
        # K = torch.sparse_coo_tensor(indices=sparseIdx, values=sK, size=(Ksize, Ksize))
        row_indices, col_indices = indices
        ## use sksparse
        K_cpu = csc_matrix((sK.detach().cpu().numpy(),
                            (row_indices, col_indices)),
                           shape=(Ksize, Ksize))
        useSksparse = True
        try:
            if useMKL:
                useSksparse = False
                factor = pyMKL.pardisoSolver((K_cpu).astype(np.float64), 2)
                factor.run_pardiso(12)
                B_cpu = factor.run_pardiso(33, f.astype(np.float64))
                if np.isnan(B_cpu[0]):
                    factor.run_pardiso(-1)
                    raise Exception("Nan happens when solving Linear system!")
            else:
                factor = cholesky(K_cpu)
                B_cpu = factor(f)

        except Exception as e:
            print(str(e))
            if useMKL:
                useSksparse = False
                factor = pyMKL.pardisoSolver((K_cpu + eye(Ksize) * 1e-7).astype(np.float64), 11)
                factor.run_pardiso(12)
                B_cpu = factor.run_pardiso(33, f.astype(np.float64))
                if np.isnan(B_cpu[0]):
                    factor.run_pardiso(-1)
                    raise Exception("Nan happens when solving Linear system after add beta@I !")

            else:
                factor = cholesky(K_cpu + eye(Ksize) * 1e-7)
                B_cpu = factor(f)

        U = torch.tensor(B_cpu, dtype=torch.float32, device=sK.device, requires_grad=False)

        ctx.factor, ctx.B_cpu, ctx.row_indices, ctx.col_indices = factor, B_cpu, row_indices, col_indices
        ctx.device = sK.device
        ctx.useSksparse = useSksparse
        return U

    @staticmethod
    def backward(ctx, grad_u):
        factor, B_cpu, row_indices, col_indices = ctx.factor, ctx.B_cpu, ctx.row_indices, ctx.col_indices
        if ctx.useSksparse:
            dSumU_dK = factor(grad_u.detach().cpu().numpy())
        else:
            # B_cpu = scipy.sparse.linalg.cg(factor, grad_u.detach().cpu().numpy(), maxiter=1000)[0]
            dSumU_dK = factor.run_pardiso(33, grad_u.detach().cpu().numpy().astype(np.float64)).copy()
            factor.run_pardiso(-1)

        dSumU_dKij = -(dSumU_dK[col_indices] * B_cpu[row_indices] + dSumU_dK[row_indices] * B_cpu[col_indices]) / 2
        dSumU_dKij_th = torch.tensor(dSumU_dKij, dtype=torch.float32, device=ctx.device, requires_grad=False)
        return dSumU_dKij_th, None, None, None, None


def sk2u(sK: torch.tensor, indices, f, f_th, Ksize):
    return Sk2Displacement.apply(sK, indices, f, f_th, Ksize)


if __name__ == '__main__':
    row = np.array([0, 0, 1, 3, 2, 2])
    col = np.array([0, 2, 1, 3, 2, 0])
    data = np.array([3, 1, 2, 1, 1, 1])
    Ksize = 4
    f = np.array([0, 1, 2, 3])
    f_th = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
    sK = torch.from_numpy(data).float()
    sK.requires_grad = True
    c = sk2c(sK, (row, col), f, f_th, 4)
    L = c * c
    L.backward()

    U = sk2u(sK, (row, col), f, f_th, 4)
    c1 = torch.dot(U, f_th)

    sU = U.sum()
    dsU_dsK = torch.autograd.grad(sU, sK)[0]
    # sU.backward()
    # c.backward()
    # print(sK.grad)
    indices = np.vstack((row, col))
    indices = torch.tensor(indices, dtype=torch.long)
    K = torch.sparse_coo_tensor(indices, sK, size=(4, 4)).to_dense()
    L = torch.linalg.cholesky(K)
    U1 = torch.cholesky_solve(f_th.unsqueeze(-1), L).squeeze(-1)
    sU1 = U1.sum()
    dsU1_dsK = torch.autograd.grad(sU1, sK)[0]
    print(dsU_dsK - dsU1_dsK)
