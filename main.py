import os
import sys

import time
import datetime

# from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torch
import numpy as np

from module.TOuNNOptimizaerFEA import TopologyOptimizer
from module.gridMesher import GridMesh

from utils.toArgparse import parser
# from utils.sftp import sftpClient

from settings.settingsFactory import problemMaterialFactory


def initComet(projName):
    experiment = Experiment(
        api_key="",
        project_name=projName,
        workspace="",
        log_code=False,
        log_graph=False,
        auto_param_logging=True,
        auto_metric_logging=True,
        parse_args=True
    )
    return experiment


class NoOp:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            if len(args):
                print(args)
            if len(kwargs):
                print(kwargs)
        return method

# generate a class that do nothing
class virtualCometExperment(NoOp):
    def __init__(self):
        print('Using the local Logger')

class sftpClient(NoOp):
    def __init__(self):
        print('Using the local Logger')
    

def main(args):
    savePathName = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M')
    cmd = sys.argv[1:]
    setattr(args, 'cmd', cmd)

    problem = problemMaterialFactory(args)

    # init comet
    # experiment = initComet(problem.name)
    experiment = virtualCometExperment()
    experiment.set_name(savePathName)

    savePath = os.path.join('data', 'results', problem.name, savePathName)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    setattr(args, 'savePath', savePath)

    # set sftp
    sftp = sftpClient()
    setattr(args, 'sftp', sftp)
    sftp.createFolderIfNotExist(savePath)

    start = time.perf_counter()
    gridMesh = GridMesh(problem)

    experiment.log_parameters(args)
    topOpt = TopologyOptimizer(problem, gridMesh, savePath, experiment, args)
    if not args.twoStage:
        topOpt.optimizeDesign(args)
    else:
        topOpt.optimizeDesign2Stage(args)

    print("Time taken (secs): {:.2F}".format(time.perf_counter() - start))

    experiment.end()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
