import numpy as np
import random
import torch
import torch.nn as nn


def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)


# %% Neural network
class TopNet(nn.Module):
    def __init__(self, nnSettings, inputDim):
        self.inputDim = inputDim  # x and y coordn of the point
        self.outputDim = 1  # if material/void at the point
        super().__init__()
        self.layers = nn.ModuleList()
        manualSeed = 1234  # NN are seeded manually
        set_seed(manualSeed)
        current_dim = self.inputDim

        for lyr in range(nnSettings['numLayers']):  # define the layers
            l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            current_dim = nnSettings['numNeuronsPerLyr']

        last_layer = nn.Linear(current_dim, self.outputDim)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        self.layers.append(last_layer)

        self.bnLayer = nn.ModuleList()
        for lyr in range(nnSettings['numLayers']):  # batch norm
            self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

    def forward(self, x):
        x1 = x
        m = nn.LeakyReLU()


        for ctr, layer in enumerate(self.layers[:-1]):  # forward prop
            if ctr % 2 == 0:
                x1 = m(self.bnLayer[ctr](layer(x1)))
                # x = self.bnLayer[ctr](layer(x)).sin()
            else:
                x1 = layer(x1).sin()
        '''
        for ctr, layer in enumerate(self.layers[:-1]):  # forward prop
            x1 = m(self.bnLayer[ctr](layer(x1)))
        '''
        output = torch.sigmoid(self.layers[-1](x1))
        rho = 0.01 + output[:, 0]
        return rho


class scalarNet(nn.Module):
    def __init__(self, nnSettings, inputDim):
        self.inputDim = inputDim  # x and y coordn of the point
        self.outputDim = 2  # if material/void at the point
        super().__init__()
        self.layers = nn.ModuleList()
        manualSeed = 1234  # NN are seeded manually
        set_seed(manualSeed)
        current_dim = self.inputDim

        for lyr in range(nnSettings['numLayers']):  # define the layers
            l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            current_dim = nnSettings['numNeuronsPerLyr']

        last_layer = nn.Linear(current_dim, self.outputDim)
        nn.init.zeros_(last_layer.bias)
        nn.init.zeros_(last_layer.weight)
        self.layers.append(last_layer)

        self.bnLayer = nn.ModuleList()
        for lyr in range(nnSettings['numLayers']):  # batch norm
            self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

    def forward(self, x):
        # x1 = x / 1000
        # x1 = x / 90
        x1 = x
        x_x, x_y, x_z = x1[:, 0] / 1e5, x1[:, 1] / 1e5, x1[:, 2] / 1e5
        # x_x, x_y, x_z = x[:, 0]/1000, x[:, 1]/1000, x[:, 2]/1000
        m = nn.SiLU()

        for ctr, layer in enumerate(self.layers[:-1]):  # forward prop
            if ctr % 2 == 0:
                x1 = m(layer(x1))
            else:
                x1 = layer(x1).tanh()

        output = torch.sigmoid(self.layers[-1](x1))
        t = output[:, 0] + x_y
        aux = output[:, 1] + x_x
        return t, aux


'''
class TopNet(nn.Module):
    def __init__(self, nnSettings, inputDim):
        self.inputDim = inputDim # x and y coordn of the point
        self.outputDim = 3 # if material/void at the point
        super().__init__()
        self.layers = nn.ModuleList()
        manualSeed = 1234 # NN are seeded manually
        set_seed(manualSeed)
        current_dim = self.inputDim
        
        for lyr in range(nnSettings['numLayers']): # define the layers
            l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            current_dim = nnSettings['numNeuronsPerLyr']

        last_layer = nn.Linear(current_dim, self.outputDim)
        nn.init.zeros_(last_layer.bias)
        nn.init.zeros_(last_layer.weight)
        self.layers.append(last_layer)

        self.bnLayer = nn.ModuleList()
        for lyr in range(nnSettings['numLayers']): # batch norm
            self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

    def forward(self, x):
        # x1 = x / 1000
        x1 = x/90
        x_x, x_y, x_z = x1[:, 0]/1000, x1[:, 1]/1000, x1[:, 2]/1000
        # x_x, x_y, x_z = x[:, 0]/1000, x[:, 1]/1000, x[:, 2]/1000
        m = nn.LeakyReLU()

        for ctr, layer in enumerate(self.layers[:-1]): # forward prop
            if ctr % 2 == 0:
                # x = m(self.bnLayer[ctr](layer(x)))
                x = self.bnLayer[ctr](layer(x)).sin()
            else:
                x = self.bnLayer[ctr](layer(x)).sin()



        output = torch.sigmoid(self.layers[-1](x1))
        rho = 0.01 + output[:, 0]
        t = output[:, 1] + x_x
        aux = output[:, 2] + x_y
        return rho, t, aux
'''
