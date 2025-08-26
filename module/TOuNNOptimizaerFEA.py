# TOuNN: Topology Optimization using Neural Networks
# Authors : Aaditya Chandrasekhar, Krishnan Suresh
# Affliation : University of Wisconsin - Madison
# Corresponding Author : ksuresh@wisc.edu , achandrasek3@wisc.edu
# Submitted to Structural and Multidisciplinary Optimization
# For academic purposes only
import json
import time

import math
import os

# %% imports
import numpy as np
import torch
import torch.optim as optim
from module.FE import FE
from module.network import TopNet, scalarNet
from diso import DiffMC
import pyvista as pv
import random
from utils.calc_triangle_normals import compute_normals


def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)

# %% main TO functionalities
class TopologyOptimizer:
    # -----------------------------#
    def __init__(self, problem, mesh, savePath, experiment, args):
        set_seed(1234)
        self.args = args
        self.savePath = savePath
        self.mesh = mesh
        self.experiment = experiment  # comet use
        self.exampleName = mesh.bc['exampleName']
        self.device = self.setDevice(args.overrideGPU)
        self.boundaryResolution = 3  # default value for plotting and interpreting
        self.FE = FE(problem, self.device)

        self.Kmax = 0
        self.deltaLayer = 0

        self.xyz = torch.tensor(self.mesh.elemCenters, requires_grad=True). \
            float().view(-1, 3).to(self.device)
        self.xyzUpSampling = torch.tensor(self.mesh.elemCentersUpSampling). \
            float().view(-1, 3).to(self.device)

        self.desiredVolumeFraction = args.desireVolumeFraction
        self.density = self.desiredVolumeFraction * np.ones((self.mesh.numElems))
        self.isotropic = args.isotropic

        densityProjection = {
            'isOn': args.densityProjection,
            'sharpness': args.densityProjectionSharpness
        }
        self.densityProjection = densityProjection

        self.fourierMap = {
            'isOn': args.fourierMap,
            'minRadius': args.fourierMapMinRadius,
            'maxRadius': args.fourierMapMaxRadius,
            'numTerms': args.fourierTerms
        }

        if (self.fourierMap['isOn']):
            coordnMap = np.zeros((3, self.fourierMap['numTerms']))
            for i in range(coordnMap.shape[0]):
                for j in range(coordnMap.shape[1]):
                    coordnMap[i, j] = np.random.choice([-1., 1.]) * np.random.uniform(
                        1. / (2 * self.fourierMap['maxRadius']), 1. / (2 * self.fourierMap['minRadius']))  #

            self.coordnMap = torch.tensor(coordnMap).float().to(self.device)  #
            inputDim = 2 * self.coordnMap.shape[1]
        else:
            self.coordnMap = torch.eye(3)
            inputDim = 3

        # inputDim = 3  # x y z coordn
        nnSettings = {'numLayers': args.numLayers, 'numNeuronsPerLyr': args.numNeuronsPerLayer}

        self.topNet = TopNet(nnSettings, inputDim).to(self.device)
        self.scalarNet = scalarNet(nnSettings, 3).to(self.device)

        self.objective = 0.

        settings_savepath = os.path.join(savePath, 'cmd.txt')
        with open(settings_savepath, 'w') as f:
            f.write(' '.join(args.cmd))
        args.sftp.save(settings_savepath)

    # -----------------------------#
    def setDevice(self, overrideGPU):
        if (torch.cuda.is_available() and (overrideGPU == False)):
            device = torch.device("cuda:0")
            print("GPU enabled")
        else:
            device = torch.device("cpu")
            print("Running on CPU")
        return device

    # -----------------------------#
    def projectDensity(self, x):
        if not self.densityProjection['isOn']:
            return x

        b = self.densityProjection['sharpness']
        nmr = np.tanh(0.5 * b) + torch.tanh(b * (x - 0.5))
        _x = 0.5 * nmr / np.tanh(0.5 * b)
        return _x

    def applyFourierMapping(self, x):
        if not self.fourierMap['isOn']:
            return x

        c = torch.cos(2 * np.pi * torch.matmul(x, self.coordnMap))
        s = torch.sin(2 * np.pi * torch.matmul(x, self.coordnMap))
        xv = torch.cat((c, s), axis=1)
        return xv

    def forcedNullElement(self, x):
        if self.mesh.nullElem is None:
            return x

        x[self.mesh.nullElem] = 0
        return x

    ## loss functions
    def loss_sr(self, objective):
        return objective / self.c0

    def loss_stress(self, objective):
        return objective / self.stress0

    def loss_vol(self, alpha, volConstraint):
        return alpha * torch.pow(volConstraint, 2)

    def loss_sf(self, mc_v, mc_f, ldp, k=10):
        mc_normal = -1 * compute_normals(mc_v, mc_f)
        mc_area = torch.linalg.norm(mc_normal, dim=1, keepdim=True)
        mc_normal_normalized = mc_normal / (mc_area + 1e-7)

        ldp_normalized = ldp / torch.linalg.norm(ldp, dim=1, keepdim=True)
        normal_dot_ldp = (mc_normal_normalized * ldp_normalized).sum(1)

        vf_center = mc_v[mc_f].mean(dim=1)
        lsf_value_ = torch.sigmoid((normal_dot_ldp + 0.70711) * -k)

        fliter_func = 0.5 * (1 / (1 + torch.exp(-k * (vf_center[:, 1] - 2)) - 0.5))
        lsf_value = lsf_value_ * fliter_func

        lsf = lsf_value.mean()
        return lsf

    def loss_curvature(self, xyz, nn_dir_lpd, nn_rho, max_curvature=0.1):
        gradT = nn_dir_lpd
        xx, yy, zz = xyz.unbind(-1)
        fx, fy, fz = nn_dir_lpd.unbind(-1)

        dfx_dxyz = \
            torch.autograd.grad(fx, xyz, torch.ones_like(fx), create_graph=True, retain_graph=True, allow_unused=True,
                                materialize_grads=True)[0]
        dfy_dxyz = \
            torch.autograd.grad(fy, xyz, torch.ones_like(fx), create_graph=True, retain_graph=True, allow_unused=True,
                                materialize_grads=True)[0]
        dfz_dxyz = \
            torch.autograd.grad(fz, xyz, torch.ones_like(fx), create_graph=True, retain_graph=True, allow_unused=True,
                                materialize_grads=True)[0]

        ddf_dxx, ddf_dxy, ddf_dxz = dfx_dxyz.unbind(-1)
        ddf_dyx, ddf_dyy, ddf_dyz = dfy_dxyz.unbind(-1)
        ddf_dzx, ddf_dzy, ddf_dzz = dfz_dxyz.unbind(-1)

        hessianMatrix = torch.stack((dfx_dxyz, dfy_dxyz, dfz_dxyz), -1)
        hessianMatrix_star = torch.stack((
            ddf_dyy * ddf_dzz - ddf_dyz * ddf_dyz,
            ddf_dyz * ddf_dxz - ddf_dxy * ddf_dzz,
            ddf_dxy * ddf_dyz - ddf_dyy * ddf_dxz,
            ddf_dyz * ddf_dxz - ddf_dxy * ddf_dzz,
            ddf_dxx * ddf_dzz - ddf_dxz * ddf_dxz,
            ddf_dxy * ddf_dxz - ddf_dxx * ddf_dyz,
            ddf_dxy * ddf_dyz - ddf_dyy * ddf_dxz,
            ddf_dxy * ddf_dxz - ddf_dxx * ddf_dyz,
            ddf_dxx * ddf_dyy - ddf_dxy * ddf_dxy,
        ), -1).reshape(xx.shape + (3, 3))

        norm_gradT = torch.norm(gradT, dim=1)

        Kg = torch.einsum('bk,bjk,bj->b', gradT, hessianMatrix_star, gradT) / (norm_gradT ** 4)
        Km = (torch.einsum('bk,bjk,bj->b', gradT, hessianMatrix, gradT) -
              torch.einsum('b, bii->b', norm_gradT ** 2, hessianMatrix)) / (norm_gradT ** 3 * 2)

        K1 = Km + (Km * Km - Kg + 1e-6).sqrt()
        # K2 = Km - (Km * Km - Kg + 1e-6).sqrt()
        Kmax = K1
        print('max:curvature  {}'.format((nn_rho * Kmax).max()))

        K1Rest = (torch.nn.functional.relu(Kmax - max_curvature))
        self.Kmax = (nn_rho * K1Rest).max()

        return (K1Rest.flatten() * nn_rho).sum() / (nn_rho.sum())
    
    def loss_fiber_curvature(self, xyz, nn_dir_fiber, nn_rho, max_curvature):
        # should be, f \nabla f \times f
        # see Curvature formulas for implicit curves and surfaces, eq 5.3
		gradT = nn_dir_fiber
		tx, ty, tz = gradT.unbind(-1)
		dt_dx = torch.autograd.grad(tx, xyz, torch.ones_like(tx), create_graph=True, retain_graph=True,
                                        allow_unused=True,materialize_grads=True)[0]
		dt_dy = torch.autograd.grad(ty, xyz, torch.ones_like(ty), create_graph=True, retain_graph=True, 
                                        allow_unused=True,materialize_grads=True)[0]
		dt_dz = torch.autograd.grad(tz, xyz, torch.ones_like(tz), create_graph=True, retain_graph=True, 
                                        allow_unused=True,materialize_grads=True)[0]
		hessianMatrix = torch.stack(([dt_dx, dt_dy, dt_dz], -1)
		
		TH = torch.einsum('bij,bj->bi', hessianMatrix, gradT)
		kappa = torch.cross(TH, gradT, dim=-1).norm(dim=-1)
		dkappa = torch.nn.functional.relu(kappa-max_curvature)
		return (dkappa * nn_rho).sum() / (nn_rho.sum() + 1e-8)
    
    def loss_gradient_length(self, nn_dir_lpd, nn_rho, c=0.001):
        gradient_length = torch.norm(nn_dir_lpd, dim=1)

        try:
            usedRho = torch.argwhere(nn_rho > 0.5).flatten()
            weighted_gradient_length = gradient_length[usedRho]
            print('max-min: layer thickness {}  -- {}'.format(weighted_gradient_length.max() / c,
                                                              weighted_gradient_length.min() / c))
        except Exception as e:
            print(str(e))

        # d = (gradient_length - c) ** 2 / c
        d = (torch.nn.functional.relu(c * 0.6 - gradient_length) + \
             torch.nn.functional.relu(gradient_length - c * 1.5)) / c
        self.deltaLayer = d.max()
        return (d.flatten() * nn_rho).sum() / nn_rho.sum()
        # return ((gradient_length - c) ** 2).mean()

    def loss_fiber_divergence(self, nn_dir_lpd, nn_dir_fiber):
        pass

    def loss_vanish(self, x, threshold=1e-1):
        d = x.norm(dim=1)
        mask = d < threshold
        barrier = torch.zeros_like(d)
        barrier[mask] = -(d[mask] - threshold) ** 2 * torch.log(d[mask] / threshold)
        return barrier.mean()

    def loss_hormonic(self, nn_dir_lpd, nn_dir_fiber, nn_rho):
        if self.args.wCurvature > 1e-9:
            lc = self.loss_curvature(self.xyz, nn_dir_lpd, nn_rho) * self.args.wCurvature
        else:
            lc = torch.tensor(0)

        lg_fiber = 0
        if self.args.wLayerThinckness > 1e-9:
            lg_lpd = self.loss_gradient_length(nn_dir_lpd, nn_rho) * self.args.wLayerThinckness
            # lg_fiber = self.loss_gradient_length(nn_dir_fiber, nn_rho, c=1e-4) * 1e-2
        else:
            lg_lpd = torch.tensor(0)
            lg_fiber = 0
        min_nn_dir_fiber = torch.linalg.norm(nn_dir_fiber, dim=1).min()
        print('lc:{} lg_lpd:{} lg_fd:{} min_nn_dir_fiber:{}'.format(lc, lg_lpd, lg_fiber, min_nn_dir_fiber))
        return lc + lg_lpd

    def loss_3axis(self, nn_dir_lpd, nn_rho, angleMax=math.pi / 6):  # less than 30^
        if self.args.w3Axis > 1e-9:
            nn_dir_lpd_normalized = nn_dir_lpd / torch.linalg.norm(nn_dir_lpd, dim=1, keepdim=True)  # (n, 3)
            nn_dir_lpd_mean = nn_dir_lpd_normalized.mean(dim=0)  # (3,)
            angleCos = nn_dir_lpd_normalized @ nn_dir_lpd_mean  # (n, 1)
            angleRelu = torch.nn.functional.relu(math.cos(angleMax) - angleCos)
            rhoWeightedAngle = (angleRelu * nn_rho / nn_rho.sum()).sum()
            return rhoWeightedAngle * self.args.w3Axis
        return torch.tensor(0)

    # -----------------------------#
    def optimizeDesign(self, args=None):
        minEpochs, maxEpochs = args.minEpoch, args.maxEpoch
        savePath = self.savePath
        sftp = args.sftp
        self.convergenceHistory = {'compliance': [], 'vol': [], 'grayElems': []}
        learningRate = args.learningRate

        alphaMax = 1000
        alphaIncrement = 0.05
        alpha = alphaIncrement  # start

        alpha1Max = 1000
        alpha1Increment = 0.05
        alpha1 = alpha1Increment

        alpha2Max = 1000
        alpha2Increment = 0.05
        alpha2 = alpha2Increment
        # alpha1 = 1 # alpha1Increment

        nrmThreshold = 0.01  # for gradient clipping
        paramaters = list(self.topNet.parameters()) + list(self.scalarNet.parameters())
        self.optimizer = optim.Adam(paramaters, amsgrad=True, lr=learningRate)
        self.scheduler = \
            optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                 min_lr=args.minLR,
                                                 factor=args.ScheduleFactor,
                                                 patience=args.SchedulePatience)

        diffmc = DiffMC(dtype=torch.float32)

        for epoch in range(maxEpochs):
            print('start optimization' + str(time.time()))

            self.optimizer.zero_grad()

            fourierXYZ = self.applyFourierMapping(self.xyz)
            nn_rho_ = self.topNet(fourierXYZ)
            nn_time_, nn_aux_ = self.scalarNet(self.xyz)

            nn_rho = self.projectDensity(nn_rho_)
            nn_rho = self.forcedNullElement(nn_rho)

            '''
            # test for the gradient \vert surface
            if epoch == 100:
                from scipy.interpolate import NearestNDInterpolator
                X = self.xyz.detach().cpu().numpy()
                X_value = nn_time_.detach().cpu().numpy()

                grid = pv.ImageData(
                    dimensions=(30, 30, 30),
                    spacing=(1, 1, 1),
                    origin=(0.5, 0.5, 0.5))

                Y = grid.points
                Y_value = NearestNDInterpolator(X, X_value)(Y)
                iso_surface = grid.contour(np.linspace(Y_value.min(), Y_value.max(), 30),
                                           Y_value,
                                           method='marching_cubes')

                pct = pv.PolyData(self.mesh.elemCenters, None)
                pct.point_data['lpd'] = nn_dir_lpd.detach().cpu().numpy()
                glyph_lpd = pct.glyph(geom=pv.Arrow(), orient='lpd', scale=False, factor=1)

                plotter = pv.Plotter()
                plotter.add_axes()
                plotter.add_mesh(glyph_lpd, color='r')
                plotter.add_mesh(iso_surface)
                plotter.show()
            '''

            nn_dir_lpd = \
                torch.autograd.grad(nn_time_, self.xyz, torch.ones_like(nn_time_),
                                    retain_graph=True, create_graph=True)[0]

            nn_dir_aux = \
                torch.autograd.grad(nn_aux_, self.xyz, torch.ones_like(nn_aux_),
                                    retain_graph=True, create_graph=True)[0]

            self.Kmax = 0
            self.deltaLayer = 0

            if args.wPlanar > 1e-9:
                args.wHarmonic = 0
                self.args.wCurvature = 0
                self.args.wLayerThinckness = 0
                self.Kmax = 0
                self.deltaLayer = 0
                nn_dir_lpd = torch.mean(nn_dir_lpd, dim=0, keepdim=True).expand_as(self.xyz)

            nn_dir_fiber = torch.cross(nn_dir_lpd, nn_dir_aux, dim=1)

            nn_dir_fiber_xz = torch.sqrt(nn_dir_fiber[:, 0] ** 2 + nn_dir_fiber[:, 2] ** 2)

            nn_theta = torch.atan2(nn_dir_fiber[:, 1], nn_dir_fiber_xz)
            nn_phi = torch.atan2(nn_dir_fiber[:, 2], nn_dir_fiber[:, 0])

            # move tensor to numpy array
            rho_np = nn_rho.cpu().detach().numpy()
            self.density = rho_np
            currentVolumeFraction = np.average(rho_np)

            # Call FE 88 line code [Niels Aage 2013]
            # objective = self.FE.solve_c_new(nn_phi, nn_theta, nn_rho, self.mesh.penal, self.isotropic)
            print('start FEA' + str(time.time()))
            stress, c = self.FE.solve_stress_new(nn_phi, nn_theta, nn_rho, self.mesh.penal, self.isotropic)
            stress_real, Criterion = self.FE.FRealMin, self.FE.Criterion
            print('End FEA' + str(time.time()))

            with torch.no_grad():
                if epoch == 0:
                    self.c0 = c.clone().detach()
                    self.stress0 = stress.clone().detach()

            volConstraint = (torch.mean(nn_rho) / self.desiredVolumeFraction) - 1.0  # global vol constraint
            self.objective = stress
            # loss = objective / self.obj0 + alpha * torch.pow(volConstraint, 2)
            loss_sr = self.loss_sr(c)
            loss_stress = self.loss_stress(stress)
            loss_vol = self.loss_vol(alpha, volConstraint)
            loss_sf = 0 * loss_sr
            loss_3axis = self.loss_3axis(nn_dir_lpd, nn_rho)

            if epoch > 10 and args.wSF > 1e-9:
                # nn_rho_grid = 1 - nn_rho.reshape(self.mesh.nelz, self.mesh.nelx, self.mesh.nely).flip(dims=[2])
                nn_rho_grid = 1 - nn_rho.reshape(self.mesh.nelz, self.mesh.nelx, self.mesh.nely)
                nn_rho_grid = nn_rho_grid.permute(1, 2, 0)
                if epoch > 50:
                    iso_value = torch.clamp_min(nn_rho_grid.mean(), 0.5)
                else:
                    iso_value = nn_rho_grid.mean()
                v, f = diffmc(nn_rho_grid - iso_value, None, device=self.device)
                v_mul = torch.tensor([[self.mesh.nelx - 0.5, 0.5 - self.mesh.nely, self.mesh.nelz - 0.5]],
                                     device=self.device).repeat([v.shape[0], 1])
                v_plus = torch.tensor([[0.5, self.mesh.nely, 0.5]], device=self.device).repeat([v.shape[0], 1])
                v = v * v_mul + v_plus
                vf_center = v[f].mean(dim=1).float()

                nn_time_new, nn_aux_new = self.scalarNet(vf_center)
                nn_dir_lpd_new = \
                    torch.autograd.grad(nn_time_new, vf_center, torch.ones_like(nn_time_new),
                                        retain_graph=True, create_graph=True)[0]
                loss_sf = self.loss_sf(mc_v=v, mc_f=f, ldp=nn_dir_lpd_new)
                # mesh = pv.make_tri_mesh(points=v.detach().cpu().numpy(), faces=f.cpu().numpy())

            loss_harmonic = self.loss_hormonic(nn_dir_lpd, nn_dir_fiber, nn_rho)
            loss = args.wSR * loss_sr + \
                   -args.wStress * loss_stress + \
                   loss_vol + \
                   args.wSF * loss_sf + \
                   args.wHarmonic * loss_harmonic * alpha1 + \
                   loss_3axis * alpha2
            # max(args.wSR, args.wStress) * loss_vol + \
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(), nrmThreshold)
            torch.nn.utils.clip_grad_norm_(self.scalarNet.parameters(), nrmThreshold)

            self.optimizer.step()
            if epoch > 100:
                self.scheduler.step(loss)

            greyElements = sum(1 for rho in rho_np if ((rho > 0.2) & (rho < 0.8)))
            relGreyElements = self.desiredVolumeFraction * greyElements / rho_np.shape[0]
            self.convergenceHistory['compliance'].append(self.objective.item())
            self.convergenceHistory['vol'].append(currentVolumeFraction)
            self.convergenceHistory['grayElems'].append(relGreyElements)

            self.mesh.penal = min(4.0, self.mesh.penal + 0.01)  # continuation scheme

            if relGreyElements < 1e-3:
                self.densityProjection['sharpness'] = min(64.0, self.densityProjection['sharpness'] + 0.01)
            if loss_vol > 1e-3:
                alpha = min(alphaMax, alpha + alphaIncrement)
            if self.Kmax > 1e-3 or self.deltaLayer > 1e-3:
                alpha1 = min(alpha1Max, alpha1 + alpha1Increment)
            if loss_3axis > 1e-3:
                alpha2 = min(alpha1Max, alpha2 + alpha2Increment)

            outputStr = "Iter {:d} , Obj {:.5F}, sf {:.5F}, harmonic {:.5F},loss {:.5F} , vol {:.5F}, grey_elem {}, lr {}". \
                format(epoch, self.objective.item(), loss_sf.detach().cpu().numpy(),
                       loss_harmonic.detach().cpu().numpy(),
                       loss.detach().cpu().numpy(), currentVolumeFraction,
                       relGreyElements, self.scheduler.get_last_lr())
            print(outputStr)

            loss_dict = {
                'loss': loss.detach().cpu().numpy(),
                'loss_stress': loss_stress.detach().cpu().numpy(),
                'loss_vol': loss_vol.detach().cpu().numpy(),
                'loss_sf': loss_sf.detach().cpu().numpy(),
                'loss_sr': loss_sr.detach().cpu().numpy(),
                'loss_harmonic': loss_harmonic.detach().cpu().numpy(),
                'loss_3axis': loss_3axis.detach().cpu().numpy(),
                'vol': torch.mean(nn_rho).detach().cpu().numpy(),
                'greyElements': relGreyElements
            }
            print(loss_dict)
            self.experiment.log_current_epoch(epoch)
            self.experiment.log_metrics(loss_dict)

            ## output
            if epoch % 50 == 0:
                self.mesh.plotField(rho_np, outputStr)
                phi_np = nn_phi.detach().cpu().numpy()
                theta_np = nn_theta.detach().cpu().numpy()
                normal_vec = np.zeros((3, theta_np.shape[0]))

                # --- ori ---
                normal_vec[0, :] = np.cos(theta_np) * np.cos(phi_np)
                normal_vec[1, :] = np.sin(theta_np)
                normal_vec[2, :] = np.cos(theta_np) * np.sin(phi_np)

                fieldPath = os.path.join(savePath, '{}.vtk'.format(epoch))
                self.mesh.saveField(rho_np, normal_vec, fieldPath)
                sftp.save(fieldPath)

                nn_rho_upsampling = self.topNet(fourierXYZ)
                nn_rho_upsampling = self.projectDensity(nn_rho_upsampling)
                nn_rho_upsampling = self.forcedNullElement(nn_rho_upsampling)

                nn_rho_upsampling = 1 - nn_rho_upsampling
                
                try:
                    nn_rho_upsampling_grid = nn_rho_upsampling.reshape(self.mesh.nelz, self.mesh.nelx, self.mesh.nely)
                    nn_rho_upsampling_grid = nn_rho_upsampling_grid.permute(1, 2, 0)
                    if epoch > 50:
                        iso_value = torch.clamp_min(nn_rho_upsampling_grid.mean(), 0.5)
                    else:
                        iso_value = nn_rho_upsampling_grid.mean()
                    v, f = diffmc(nn_rho_upsampling_grid - iso_value, None, device=self.device)
                    v_mul = torch.tensor([[self.mesh.nelx - 0.5, 0.5 - self.mesh.nely, self.mesh.nelz - 0.5]],
                                         device=self.device).repeat([v.shape[0], 1])
                    v_plus = torch.tensor([[0.25, self.mesh.nely, 0.25]], device=self.device).repeat([v.shape[0], 1])
                    v = v * v_mul + v_plus
                    v_np = v.detach().cpu().numpy()

                    mesh = pv.make_tri_mesh(points=v_np, faces=f.cpu().numpy())

                    if mesh.n_points > 0:
                        mesh_savepath = os.path.join(savePath, 'MCResult_{}.obj').format(epoch)
                        pv.save_meshio(mesh_savepath, mesh)
                        sftp.save(mesh_savepath)
                except Exception as e:
                    print(str(e))

                # if epoch ==50 or epoch ==100 or epoch==200:
                pct = pv.PolyData(self.mesh.elemCenters, None)

                pct.point_data['density'] = rho_np
                pct.point_data['lpd'] = nn_dir_lpd.detach().cpu().numpy()
                pct.point_data['fiber'] = nn_dir_fiber.detach().cpu().numpy()
                pct.point_data['t'] = nn_time_.detach().cpu().numpy()

                pct.point_data['Criterion'] = Criterion.detach().cpu().numpy()

                pct_savepath = os.path.join(savePath, 'pct_{}.vtk').format(epoch)
                pct.save(pct_savepath)
                sftp.save(pct_savepath)

                pth_savepath = os.path.join(savePath, '{}.pth').format(epoch)
                torch.save({'scalarNet_state_dict': self.scalarNet.state_dict(),
                            'topNet_state_dict': self.topNet.state_dict()},
                           pth_savepath)
                sftp.save(pth_savepath)

        return self.convergenceHistory