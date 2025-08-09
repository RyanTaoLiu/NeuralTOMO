import argparse
import datetime

parser = argparse.ArgumentParser(description='argparse of Neural Topology Optimization'
                                             ' for Multi-axis Additive Manufacturing')

# general settings
parser.add_argument('--problem', type=str, required=True, help='experiment name')
parser.add_argument('--material', type=str, default='', help='material, \'\'(use problem given material)/PLA')

parser.add_argument('--desireVolumeFraction', type=float, default=0.25, help='desire Volume Fraction')
parser.add_argument('--overrideGPU', action=argparse.BooleanOptionalAction, default=False, help='Ignored Cuda')
parser.add_argument('--twoStage', action=argparse.BooleanOptionalAction, default=False, help='2-Stage Optimization')


parser.add_argument('--pdeSolver', choices=['PINN', 'FEA'], default='FEA', help='pde Solver, \'PINN\' or \'FEA\'(default)')
parser.add_argument('--minEpoch', type=int, default=2000, help='min iteration times')
parser.add_argument('--tol', type=float, default=1e-6, help='min iteration times')
parser.add_argument('--maxEpoch', type=int, default=5000, help='max iteration times')

parser.add_argument('--isotropic', action=argparse.BooleanOptionalAction, default=False, help='calculate thr direction of fiber')

# weight
parser.add_argument('--wSF', type=float, default=1, help='weight for support free')
parser.add_argument('--wSF_alpha', type=float, default=45, help='support free angle defalut(45 degree)')
parser.add_argument('--wSR', type=float, default=1, help='weight for reinforcement(compliance)')
parser.add_argument('--wStress', type=float, default=1, help='weight for reinforcement(max stress)')

## weight->wHarmonic
parser.add_argument('--wHarmonic', type=float, default=1, help='weight for harmonic')
parser.add_argument('--wCurvature', type=float, default=1, help='weight for curvature')
parser.add_argument('--wLayerThinckness', type=float, default=1, help='weight for layer thinckness')
parser.add_argument('--wSpecificSurfaceArea', type=float, default=0, help='weight for specific surface area')
parser.add_argument('--wLayerArea', type=float, default=0, help='weight for layer area')

## weight-> 3axis
parser.add_argument('--w3Axis', type=float, default=0, help='weight for 3axis')
parser.add_argument('--wPlanar', type=float, default=0, help='weight for planar')


# Network
parser.add_argument('--numLayers', type=int, default=5, help='NN layers')
parser.add_argument('--numNeuronsPerLayer', type=int, default=50, help='NN number of neurons per layer')

parser.add_argument('--learningRate', type=float, default=0.001, help='learning rate')
parser.add_argument('--minLR', type=float, default=1e-5, help='min learning rate')
parser.add_argument('--ScheduleFactor', type=float, default=0.5, help='ReduceLROnPlateau factor')
parser.add_argument('--SchedulePatience', type=float, default=20, help='ReduceLROnPlateau patience')


# density projection
parser.add_argument('--densityProjection', action=argparse.BooleanOptionalAction, default=True, help='use density projection')
parser.add_argument('--densityProjectionSharpness', type=float, default=8, help='density projection sharpness')

# Four
parser.add_argument('--fourierMap', action=argparse.BooleanOptionalAction, default=False, help='use fourier Map encode position')
parser.add_argument('--fourierMapMinRadius', type=float, default=6, help='fourier map Min radius')
parser.add_argument('--fourierMapMaxRadius', type=float, default=30, help='fourier map Max radius')
parser.add_argument('--fourierTerms', type=int, default=150, help='fourier map terms')

