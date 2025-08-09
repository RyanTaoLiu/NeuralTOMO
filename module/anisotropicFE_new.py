import math

import numpy as np
import torch

class H8_isotropic_K:
    pass

class H8_anisotropic_K:
    def __init__(self, device=torch.device('cuda'), **kwargs):
        # for const number
        _sqrt_3_5 = math.sqrt(3 / 5)
        _E_f = 1 if 'Ef' not in kwargs else kwargs['Ef']
        _E_t = 1 if 'Et' not in kwargs else kwargs['Et']
        _nu_f = 0.3 if 'nuf' not in kwargs else kwargs['nuf']
        _nu_t = 0.3 if 'nut' not in kwargs else kwargs['nut']

        if 'P' not in kwargs:
            self.P = torch.tensor([[ 0.00126103,  0.00017645, -0.00143748,  0.        ,  0.        , 0.        ],
                                    [ 0.00017645,  0.00265957, -0.00283602,  0.        ,  0.        ,0.        ],
                                    [-0.00143748, -0.00283602,  0.0042735 ,  0.        ,  0.        ,0.        ],
                                    [ 0.        ,  0.        ,  0.        ,  0.03125   ,  0.        ,0.        ],
                                    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.03125   ,0.        ],
                                    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,0.03125   ]],
                                  dtype=torch.float32, device=device)
            self.Q = torch.tensor([0.00283733, 0.0099734 , 0.03632479, 0.        , 0.        ,0.        ],
                                  dtype=torch.float32, device=device)

        else:
            self.P = torch.tensor(kwargs['P'], dtype=torch.float32, device=device)
            self.Q = torch.tensor(kwargs['Q'], dtype=torch.float32, device=device)

        self.Ef = _E_f
        self.Et = _E_t
        self.nuf = _nu_f
        self.nut = _nu_t

        _G_t = _E_t / (2 * (1 + _nu_t))
        _G_f = _E_f / (2 * (1 + _nu_f))

        C = np.array([[1 / _E_f, -_nu_f / _E_f, -_nu_f / _E_f, 0, 0, 0],
                      [-_nu_f / _E_f, 1 / _E_t, -_nu_t / _E_t, 0, 0, 0],
                      [-_nu_f / _E_f, -_nu_t / _E_t, 1 / _E_t, 0, 0, 0],
                      [0, 0, 0, 1 / _G_t, 0, 0],
                      [0, 0, 0, 0, 1 / _G_f, 0],
                      [0, 0, 0, 0, 0, 1 / _G_f]])
        self.C_inv_np = np.linalg.inv(C)
        self.C_inv = torch.tensor(self.C_inv_np, dtype=torch.float32, device=device)

        # 3 - point Gauss integration
        integration_point = torch.tensor([-_sqrt_3_5, 0, _sqrt_3_5],
                                         dtype=torch.float32) / 2
        integration_weight = torch.tensor([5 / 9, 8 / 9, 5 / 9],
                                          dtype=torch.float32) / 2

        all_intergration_points = np.vstack(
            np.meshgrid(integration_point, integration_point, integration_point)).reshape(3, -1).T

        all_intergration_weight_temp = np.vstack(
            np.meshgrid(integration_weight, integration_weight, integration_weight)).reshape(3, -1).T

        int_weight = all_intergration_weight_temp[:, 0] * \
                     all_intergration_weight_temp[:, 1] * \
                     all_intergration_weight_temp[:, 2]
        self.int_weight = torch.tensor(int_weight, device=device)

        # B = np.einsum('i,ijk->ijk', all_intergration_weight, self.matrixB(all_intergration_points))
        # [x,y,z,zy,zx,yx]
        self.B = torch.tensor(self.matrixB(all_intergration_points), dtype=torch.float32, device=device)
        self.NodeB = torch.tensor([
        [-0.25,     0,     0,  0.25,     0,     0,  0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0, 0.25,    0,    0, -0.25,     0,     0],
        [    0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0,  0.25,     0,     0, -0.25,     0,     0, -0.25,     0,    0, 0.25,    0,     0,  0.25,     0],
        [    0,     0, -0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0,  0.25,    0,    0, 0.25,     0,     0,  0.25],
        [    0, -0.25, -0.25,     0, -0.25, -0.25,     0, -0.25,  0.25,     0, -0.25,  0.25,     0,  0.25, -0.25,     0,  0.25, -0.25,    0, 0.25, 0.25,     0,  0.25,  0.25],
        [-0.25,     0, -0.25, -0.25,     0,  0.25, -0.25,     0,  0.25, -0.25,     0, -0.25,  0.25,     0, -0.25,  0.25,     0,  0.25, 0.25,    0, 0.25,  0.25,     0, -0.25],
        [-0.25, -0.25,     0, -0.25,  0.25,     0,  0.25,  0.25,     0,  0.25, -0.25,     0, -0.25, -0.25,     0, -0.25,  0.25,     0, 0.25, 0.25,    0,  0.25, -0.25,     0]],
        dtype=torch.float32, device=device)

    def matrixB(self, xyz):
        # [x,y,z,zy,zx,yx]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # x, y, z = torch.unbind(xyz, -1)
        o = np.zeros_like(x)
        b = (-(0.5 - y) * (0.5 - z), o, o, (0.5 - y) * (0.5 - z), o, o,
             (0.5 - z) * (y + 0.5), o, o, -(0.5 - z) * (y + 0.5), o, o,
             -(0.5 - y) * (z + 0.5), o, o, (0.5 - y) * (z + 0.5), o, o,
             (y + 0.5) * (z + 0.5), o, o, -(y + 0.5) * (z + 0.5), o, o,
             o, -(0.5 - x) * (0.5 - z), o, o, -(0.5 - z) * (x + 0.5), o,
             o, (0.5 - z) * (x + 0.5), o, o, (0.5 - x) * (0.5 - z), o,
             o, -(0.5 - x) * (z + 0.5), o, o, -(x + 0.5) * (z + 0.5), o,
             o, (x + 0.5) * (z + 0.5), o, o, (0.5 - x) * (z + 0.5), o,
             o, o, -(0.5 - x) * (0.5 - y), o, o, -(0.5 - y) * (x + 0.5),
             o, o, -(x + 0.5) * (y + 0.5), o, o, -(0.5 - x) * (y + 0.5),
             o, o, (0.5 - x) * (0.5 - y), o, o, (0.5 - y) * (x + 0.5),
             o, o, (x + 0.5) * (y + 0.5), o, o, (0.5 - x) * (y + 0.5),
             o, -(0.5 - x) * (0.5 - y), -(0.5 - x) * (0.5 - z), o, -(0.5 - y) * (x + 0.5), -(0.5 - z) * (x + 0.5),
             o, -(x + 0.5) * (y + 0.5), (0.5 - z) * (x + 0.5), o, -(0.5 - x) * (y + 0.5), (0.5 - x) * (0.5 - z),
             o, (0.5 - x) * (0.5 - y), -(0.5 - x) * (z + 0.5), o, (0.5 - y) * (x + 0.5), -(x + 0.5) * (z + 0.5),
             o, (x + 0.5) * (y + 0.5), (x + 0.5) * (z + 0.5), o, (0.5 - x) * (y + 0.5), (0.5 - x) * (z + 0.5),
             -(0.5 - x) * (0.5 - y), o, -(0.5 - y) * (0.5 - z), -(0.5 - y) * (x + 0.5), o, (0.5 - y) * (0.5 - z),
             -(x + 0.5) * (y + 0.5), o, (0.5 - z) * (y + 0.5), -(0.5 - x) * (y + 0.5), o, -(0.5 - z) * (y + 0.5),
             (0.5 - x) * (0.5 - y), o, -(0.5 - y) * (z + 0.5), (0.5 - y) * (x + 0.5), o, (0.5 - y) * (z + 0.5),
             (x + 0.5) * (y + 0.5), o, (y + 0.5) * (z + 0.5), (0.5 - x) * (y + 0.5), o, -(y + 0.5) * (z + 0.5),
             -(0.5 - x) * (0.5 - z), -(0.5 - y) * (0.5 - z), o, -(0.5 - z) * (x + 0.5), (0.5 - y) * (0.5 - z), o,
             (0.5 - z) * (x + 0.5), (0.5 - z) * (y + 0.5), o, (0.5 - x) * (0.5 - z), -(0.5 - z) * (y + 0.5), o,
             -(0.5 - x) * (z + 0.5), -(0.5 - y) * (z + 0.5), o, -(x + 0.5) * (z + 0.5), (0.5 - y) * (z + 0.5), o,
             (x + 0.5) * (z + 0.5), (y + 0.5) * (z + 0.5), o, (0.5 - x) * (z + 0.5), -(y + 0.5) * (z + 0.5), o)
        return np.stack(b, -1).reshape((xyz.shape[0], 6, 24))

    def angle2Ke(self, phi, theta, density, density_penal=3):
        cosT, sinT = torch.cos(theta), torch.sin(theta)
        cosT2, sinT2 = cosT * cosT, sinT * sinT

        cosP, sinP = torch.cos(phi), torch.sin(phi)
        cosP2, sinP2 = cosP * cosP, sinP * sinP

        o = torch.zeros_like(phi) # 0-vector

        R = torch.stack((
            cosP2 * cosT2, cosP2 * sinT2, sinP2, 2 * cosP * sinP * sinT, -2 * cosP * cosT * sinP, -2 * cosP2 * cosT * sinT,
            sinT2, cosT2, o, o, o, 2 * cosT * sinT,
            cosT2 * sinP2, sinP2 * sinT2, cosP2, -2 * cosP * sinP * sinT, 2 * cosP * cosT * sinP,
            -2 * cosT * sinP2 * sinT,
            cosT * sinP * sinT, -cosT * sinP * sinT, o, cosP * cosT, cosP * sinT, sinP * (cosT2 - sinT2),
            cosP * cosT2 * sinP, cosP * sinP * sinT2, -cosP * sinP, -sinT * (cosP2 - sinP2),
            cosT * (cosP2 - sinP2), -2 * cosP * cosT * sinP * sinT,
            cosP * cosT * sinT, -cosP * cosT * sinT, o, -cosT * sinP, -sinP * sinT, cosP * (cosT2 - sinT2)
        ), -1).reshape(phi.shape + (6, 6))

        # C_inv 6x6
        # C = R @ self.C_inv.unsqueeze(0).expand(batch_size, -1, -1) @ R.transpose(1, 2)

        C = torch.einsum('bij,jk,blk->bil', R, self.C_inv, R)
        C_new = torch.einsum('bji,bjk->bik', R, C)

        # self.B 27x6x24, C nx6x6
        B = self.B
        weight = self.int_weight
        # BT = B.transpose(1, 2)

        BT_C_B = torch.einsum('d,dji,bjk,dkl->bil', weight, B, C, B)
        # dK = weight * BT.expand(batch_size, -1, -1) * C * B.expand(batch_size, -1, -1)
        dK = torch.einsum('i,ijk->ijk', (1e-3 + density) ** density_penal, BT_C_B)
        self.temp_C = C_new
        self.T = R
        return dK


# via rotation Matrix
class H8_anisotropic_K_R:
    def __init__(self, device=torch.device('cuda'), **kwargs):
        # for const number
        _sqrt_3_5 = math.sqrt(3 / 5)
        s, t, r = 0.5, 0.5, 0.5
        _E_f = 5 if 'Ef' not in kwargs else kwargs['Ef']
        _E_t = 1 if 'Et' not in kwargs else kwargs['Et']
        _nu_f = 0.3 if 'nuf' not in kwargs else kwargs['nuf']
        _nu_t = 0.32 if 'nut' not in kwargs else kwargs['nut']

        self.Ef = _E_f
        self.Et = _E_t
        self.nuf = _nu_f
        self.nut = _nu_t

        _G_t = _E_t / (2 * (1 + _nu_t))
        _G_f = _E_f / (2 * (1 + _nu_f))

        C = np.array([[1 / _E_f, -_nu_f / _E_f, -_nu_f / _E_f, 0, 0, 0],
                      [-_nu_f / _E_f, 1 / _E_t, -_nu_t / _E_t, 0, 0, 0],
                      [-_nu_f / _E_f, -_nu_t / _E_t, 1 / _E_t, 0, 0, 0],
                      [0, 0, 0, 1 / _G_t, 0, 0],
                      [0, 0, 0, 0, 1 / _G_f, 0],
                      [0, 0, 0, 0, 0, 1 / _G_f]])
        self.C_inv_np = np.linalg.inv(C)
        self.C_inv = torch.tensor(self.C_inv_np, dtype=torch.float32, device=device)

        # 3 - point Gauss integration
        integration_point = torch.tensor([-_sqrt_3_5, 0, _sqrt_3_5],
                                         dtype=torch.float32) / 2
        integration_weight = torch.tensor([5 / 9, 8 / 9, 5 / 9],
                                          dtype=torch.float32) / 2

        all_intergration_points = np.vstack(
            np.meshgrid(integration_point, integration_point, integration_point)).reshape(3, -1).T
        all_intergration_weight_temp = np.vstack(
            np.meshgrid(integration_weight, integration_weight, integration_weight)).reshape(3, -1).T
        int_weight = all_intergration_weight_temp[:, 0] * \
                     all_intergration_weight_temp[:, 1] * \
                     all_intergration_weight_temp[:, 2]
        self.int_weight = torch.tensor(int_weight, device=device)
        # B = np.einsum('i,ijk->ijk', all_intergration_weight, self.matrixB(all_intergration_points))
        self.B = torch.tensor(self.matrixB(all_intergration_points), dtype=torch.float32, device=device)

    def matrixB(self, xyz):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # x, y, z = torch.unbind(xyz, -1)
        o = np.zeros_like(x)
        b = (-(0.5 - y) * (0.5 - z), o, o, (0.5 - y) * (0.5 - z), o, o,
             (0.5 - z) * (y + 0.5), o, o, -(0.5 - z) * (y + 0.5), o, o,
             -(0.5 - y) * (z + 0.5), o, o, (0.5 - y) * (z + 0.5), o, o,
             (y + 0.5) * (z + 0.5), o, o, -(y + 0.5) * (z + 0.5), o, o,
             o, -(0.5 - x) * (0.5 - z), o, o, -(0.5 - z) * (x + 0.5), o,
             o, (0.5 - z) * (x + 0.5), o, o, (0.5 - x) * (0.5 - z), o,
             o, -(0.5 - x) * (z + 0.5), o, o, -(x + 0.5) * (z + 0.5), o,
             o, (x + 0.5) * (z + 0.5), o, o, (0.5 - x) * (z + 0.5), o,
             o, o, -(0.5 - x) * (0.5 - y), o, o, -(0.5 - y) * (x + 0.5),
             o, o, -(x + 0.5) * (y + 0.5), o, o, -(0.5 - x) * (y + 0.5),
             o, o, (0.5 - x) * (0.5 - y), o, o, (0.5 - y) * (x + 0.5),
             o, o, (x + 0.5) * (y + 0.5), o, o, (0.5 - x) * (y + 0.5),
             o, -(0.5 - x) * (0.5 - y), -(0.5 - x) * (0.5 - z), o, -(0.5 - y) * (x + 0.5), -(0.5 - z) * (x + 0.5),
             o, -(x + 0.5) * (y + 0.5), (0.5 - z) * (x + 0.5), o, -(0.5 - x) * (y + 0.5), (0.5 - x) * (0.5 - z),
             o, (0.5 - x) * (0.5 - y), -(0.5 - x) * (z + 0.5), o, (0.5 - y) * (x + 0.5), -(x + 0.5) * (z + 0.5),
             o, (x + 0.5) * (y + 0.5), (x + 0.5) * (z + 0.5), o, (0.5 - x) * (y + 0.5), (0.5 - x) * (z + 0.5),
             -(0.5 - x) * (0.5 - y), o, -(0.5 - y) * (0.5 - z), -(0.5 - y) * (x + 0.5), o, (0.5 - y) * (0.5 - z),
             -(x + 0.5) * (y + 0.5), o, (0.5 - z) * (y + 0.5), -(0.5 - x) * (y + 0.5), o, -(0.5 - z) * (y + 0.5),
             (0.5 - x) * (0.5 - y), o, -(0.5 - y) * (z + 0.5), (0.5 - y) * (x + 0.5), o, (0.5 - y) * (z + 0.5),
             (x + 0.5) * (y + 0.5), o, (y + 0.5) * (z + 0.5), (0.5 - x) * (y + 0.5), o, -(y + 0.5) * (z + 0.5),
             -(0.5 - x) * (0.5 - z), -(0.5 - y) * (0.5 - z), o, -(0.5 - z) * (x + 0.5), (0.5 - y) * (0.5 - z), o,
             (0.5 - z) * (x + 0.5), (0.5 - z) * (y + 0.5), o, (0.5 - x) * (0.5 - z), -(0.5 - z) * (y + 0.5), o,
             -(0.5 - x) * (z + 0.5), -(0.5 - y) * (z + 0.5), o, -(x + 0.5) * (z + 0.5), (0.5 - y) * (z + 0.5), o,
             (x + 0.5) * (z + 0.5), (y + 0.5) * (z + 0.5), o, (0.5 - x) * (z + 0.5), -(y + 0.5) * (z + 0.5), o)
        return np.stack(b, -1).reshape((xyz.shape[0], 6, 24))

    def angle2KeExt(self, vol_ratio, rotationMatrix, density, density_penal=3):
        EF = vol_ratio * self.Ef + (1 - vol_ratio) * self.Et
        ET = 1. / (vol_ratio / self.Ef + (1 - vol_ratio) / self.Et)
        _nu_f, _nu_t = self.nuf, self.nut

        G_f = EF / (2 * (1 + _nu_f))
        G_t = ET / (2 * (1 + _nu_t))
        o = torch.zeros_like(vol_ratio)

        C = torch.stack((1 / EF, -_nu_f / EF, -_nu_f / EF, o, o, o,
                         -_nu_f / EF, 1 / ET, -_nu_t / ET, o, o, o,
                         -_nu_f / EF, -_nu_t / ET, 1 / ET, o, o, o,
                         o, o, o, 1 / G_t, o, o,
                         o, o, o, o, 1 / G_f, o,
                         o, o, o, o, o, 1 / G_f), -1).reshape(vol_ratio.shape + (6, 6))
        C_inv = torch.inverse(C)
        R = rotationMatrix
        T = torch.stack((
            R[0, 0] ** 2, R[0, 1] ** 2, R[0, 2] ** 2, 2 * R[0, 1] * R[0, 2], 2 * R[0, 0] * R[0, 2], 2 * R[0, 0] * R[0, 1],
            R[1, 0] ** 2, R[1, 1] ** 2, R[1, 2] ** 2, 2 * R[1, 1] * R[1, 2], 2 * R[1, 0] * R[1, 2], 2 * R[1, 0] * R[1, 1],
            R[2, 0] ** 2, R[2, 1] ** 2, R[2, 2] ** 2, 2 * R[2, 1] * R[2, 2], 2 * R[2, 0] * R[2, 2], 2 * R[2, 0] * R[2, 1],
            R[0, 0] * R[1, 0], R[0, 1] * R[1, 1], R[0, 2] * R[1, 2], R[0, 1] * R[1, 2] + R[0, 2] * R[1, 1], R[0, 0] * R[1, 2] + R[0, 2] * R[1, 0], R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0],
            R[1, 0] * R[2, 0], R[1, 1] * R[2, 1], R[1, 2] * R[2, 2], R[1, 1] * R[2, 2] + R[1, 2] * R[2, 1], R[1, 0] * R[2, 2] + R[1, 2] * R[2, 0], R[1, 0] * R[2, 1] + R[1, 1] * R[2, 0],
            R[0, 0] * R[2, 0], R[0, 1] * R[2, 1], R[0, 2] * R[2, 2], R[0, 1] * R[2, 2] + R[0, 2] * R[2, 1], R[0, 0] * R[2, 2] + R[0, 2] * R[2, 0], R[0, 0] * R[2, 1] + R[0, 1] * R[2, 0],
        ), -1).reshape(6, 6)

        RC = torch.einsum('bij,jk,blk->bil', T, C_inv, T)

        weight, B = self.int_weight, self.B
        BT_C_B = torch.einsum('d,dji,bjk,dkl->bil', weight, B, RC, B)
        dK = torch.einsum('i,ijk->ijk', (1e-3 + density) ** density_penal, BT_C_B)
        return dK

if __name__ == '__main__':
    K = H8_anisotropic_K()

    phi = torch.linspace(0, torch.pi * 2, 100).cuda()
    theta = torch.zeros(100).cuda()
    density = torch.zeros(100, dtype=torch.float32).cuda() + 1
    L = K.angle2Ke(phi, theta, density)
    L9 = L[9].detach().cpu().numpy()
    # map = [18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
    # LMap = L[:, map, :][:, :, map]
    # L9 = LMap[9].detach().cpu().numpy()
    print('sinT:{}, cosT:{}, sinP:{}, cosP:{}'.format(math.sin(theta[9]), math.cos(theta[9]),
                                                      math.sin(phi[9]), math.cos(phi[9])))
    print(L)
    # need to check L[99] == matlab code result
