from abc import ABC
import os
import numpy as np
from scipy.io import loadmat
import inspect


class problemBase(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

        self.mesh = None
        self.boundaryCondition = None
        self.materialProperty = None

        self.ext = None # file Extensions

    def serialize(self, savePath=None):
        outDict = {
            'name': self.name,
            'mesh': self.mesh,
            'boundaryCondition': self.boundaryCondition,
            'materialProperty': self.materialProperty
        }
        if savePath:
            np.save(savePath, outDict)
        else:
            print(outDict)

    @classmethod
    def unserialize(cls ,savePath: str):
        return npyProblem(savePath)


class npyProblem(problemBase):
    ext = 'npy'
    def __init__(self, savePath):
        super().__init__()
        self.ext = 'npy'
        data = np.load(savePath, allow_pickle=True)
        self.name = data.item().get('name')
        self.mesh = data.item().get('mesh')
        self.boundaryCondition = data.item().get('boundaryCondition')
        self.materialProperty = data.item().get('materialProperty')

class matProblem(problemBase):
    ext = 'mat'
    def __init__(self, savePath):
        super().__init__()
        data = loadmat(savePath)
        self.ext = 'mat'
        self.name = data['name']
        self.mesh = data['mesh']
        self.boundaryCondition = data['boundaryCondition']
        self.materialProperty = data['materialProperty']

