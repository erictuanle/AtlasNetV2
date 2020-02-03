from __future__  import print_function
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
import torchvision
import os

#TODO dim -> input dim output dim

class linearTransformMLP(nn.Module):
    """linear transformation module"""

    def __init__(self, atlasnet_nlatent = 1024):
        super(linearTransformMLP, self).__init__()

        self.conv1 = torch.nn.Conv1d(atlasnet_nlatent, atlasnet_nlatent//2, 1)
        self.conv2 = torch.nn.Conv1d(atlasnet_nlatent//2, atlasnet_nlatent//2, 1)
        self.conv3 = torch.nn.Conv1d(atlasnet_nlatent//2, 16, 1)
        self.bn1 = torch.nn.BatchNorm1d(atlasnet_nlatent//2)
        self.bn2 = torch.nn.BatchNorm1d(atlasnet_nlatent//2)
        self.th = nn.Tanh()

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.th(self.conv3(x))
        x = x.view(x.size(0),4,4).contiguous()
        return x

class linearAdj(nn.Module):
    """ prediction the linear transdormation matrix"""

    def __init__(self, dim = 3,atlasnet_nlatent = 1024):
        super(linearAdj, self).__init__()

        self.conv1 = torch.nn.Conv1d(atlasnet_nlatent, atlasnet_nlatent//2, 1)
        self.conv2 = torch.nn.Conv1d(atlasnet_nlatent//2, atlasnet_nlatent//2, 1)
        self.conv3 = torch.nn.Conv1d(atlasnet_nlatent//2, (dim+1)*3, 1)
        self.bn1 = torch.nn.BatchNorm1d(atlasnet_nlatent//2)
        self.bn2 = torch.nn.BatchNorm1d(atlasnet_nlatent//2)
        self.th = nn.Tanh()
        self.dim = dim

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.th(self.conv3(x))
        R = x[:,0:self.dim*3].view(x.size(0),self.dim,3).contiguous()
        T = x[:,self.dim*3:].view(x.size(0),1,3).contiguous()
        return R,T

class mlpAdj(nn.Module):
    def __init__(self, atlasnet_nlatent = 1024):
        """Atlas decoder"""

        super(mlpAdj, self).__init__()
        self.atlasnet_nlatent = atlasnet_nlatent
        self.conv1 = torch.nn.Conv1d(self.atlasnet_nlatent, self.atlasnet_nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.atlasnet_nlatent, self.atlasnet_nlatent//2, 1)
        self.conv3 = torch.nn.Conv1d(self.atlasnet_nlatent//2, self.atlasnet_nlatent//4, 1)
        self.conv4 = torch.nn.Conv1d(self.atlasnet_nlatent//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.atlasnet_nlatent)
        self.bn2 = torch.nn.BatchNorm1d(self.atlasnet_nlatent//2)
        self.bn3 = torch.nn.BatchNorm1d(self.atlasnet_nlatent//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class patchDeformationMLP(nn.Module):
    """deformation of a 2D patch into a 3D surface"""

    def __init__(self,patchDim=2,patchDeformDim=3,tanh=True):

        super(patchDeformationMLP, self).__init__()
        layer_size = 128
        self.tanh=tanh
        self.conv1 = torch.nn.Conv1d(patchDim, layer_size, 1)
        self.conv2 = torch.nn.Conv1d(layer_size, layer_size, 1)
        self.conv3 = torch.nn.Conv1d(layer_size, patchDeformDim, 1)
        self.bn1 = torch.nn.BatchNorm1d(layer_size)
        self.bn2 = torch.nn.BatchNorm1d(layer_size)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.tanh:
            x = self.th(self.conv3(x))
        else:
            x = self.conv3(x)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, npoint = 2500, atlasnet_nlatent = 1024):
        """Encoder"""

        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, atlasnet_nlatent, 1)
        self.lin = nn.Linear(atlasnet_nlatent, atlasnet_nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(atlasnet_nlatent)
        self.bn4 = torch.nn.BatchNorm1d(atlasnet_nlatent)

        self.npoint = npoint
        self.atlasnet_nlatent = atlasnet_nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, self.atlasnet_nlatent)
        x = F.relu(self.bn4(self.lin(x).unsqueeze(-1)))
        return x[...,0]

class AtlasNet(nn.Module):
    """Atlas net auto encoder"""

    def __init__(self, options):

        super(AtlasNet, self).__init__()

        self.npoint = options.npoint
        self.nb_patches = options.nb_patches
        self.atlasnet_nlatent = options.atlasnet_nlatent
        self.patchDim = options.patchDim

        #encoder and decoder modules
        #==============================================================================
        self.encoder = PointNetfeat(self.npoint,self.atlasnet_nlatent)
        self.decoder = nn.ModuleList([mlpAdj(atlasnet_nlatent = 2 +self.atlasnet_nlatent) for i in range(0,self.nb_patches)])
        #==============================================================================

    def forward(self, x):

        #encoder
        #==============================================================================
        x = self.encoder(x.transpose(2,1).contiguous())
        #==============================================================================

        outs = []
        patches = []
        for i in range(0,self.nb_patches):

            #random patch
            #==========================================================================
            rand_grid = torch.FloatTensor(x.size(0),self.patchDim,self.npoint//self.nb_patches).cuda()
            rand_grid.data.uniform_(0,1)
            rand_grid[:,2:,:] = 0
            patches.append(rand_grid[0].transpose(1,0))
            #==========================================================================

            #cat with latent vector and decode
            #==========================================================================
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
            #==========================================================================

        return torch.cat(outs,2).transpose(2,1).contiguous(), patches


class AtlasNetLinAdj(nn.Module):
    """Atlas net auto encoder"""

    def __init__(self, options):

        super(AtlasNetLinAdj, self).__init__()

        self.npoint = options.npoint
        self.nb_patches = options.nb_patches
        self.atlasnet_nlatent = options.atlasnet_nlatent
        self.patchDim = options.patchDim

        #encoder and decoder modules
        #==============================================================================
        self.encoder = PointNetfeat(self.npoint,self.atlasnet_nlatent)
        self.linearTransformMatrix = nn.ModuleList(linearAdj(dim=self.patchDim,atlasnet_nlatent=self.atlasnet_nlatent) for i in range(0,self.nb_patches))
        #==============================================================================

    def forward(self, x):

        #encoder
        #==============================================================================
        x = self.encoder(x.transpose(2,1).contiguous())
        #==============================================================================

        outs = []
        patches = []
        for i in range(0,self.nb_patches):

            #random patch
            #==========================================================================
            rand_grid = torch.FloatTensor(x.size(0),self.patchDim,self.npoint//self.nb_patches).cuda()
            rand_grid.data.uniform_(0,1)
            rand_grid[:,2:,:] = 0
            patches.append(rand_grid[0].transpose(1,0))
            #==========================================================================

            #cat with latent vector and decode
            #==========================================================================
            R,T = self.linearTransformMatrix[i](x.unsqueeze(2))
            rand_grid = torch.bmm(rand_grid.transpose(2,1),R) + T
            outs.append(rand_grid)
            #==========================================================================

        return torch.cat(outs,2).transpose(2,1).contiguous(), patches


class PointTransMLPAdj(nn.Module):
    """Atlas net auto encoder"""

    def __init__(self, options):

        super(PointTransMLPAdj, self).__init__()

        self.npoint = options.npoint
        self.nb_patches = options.nb_patches
        self.atlasnet_nlatent = options.atlasnet_nlatent
        self.nbatch = options.nbatch
        self.dim = options.patchDim

        #encoder and decoder modules
        #==============================================================================
        self.encoder = PointNetfeat(self.npoint,self.atlasnet_nlatent)
        self.decoder = nn.ModuleList([mlpAdj(atlasnet_nlatent = self.dim + self.atlasnet_nlatent) for i in range(0,self.nb_patches)])
        #==============================================================================

        #patch
        #==============================================================================
        self.grid = []
        for patchIndex in range(self.nb_patches):
            patch = torch.nn.Parameter(torch.FloatTensor(1,self.dim,self.npoint//self.nb_patches))
            patch.data.uniform_(0,1)
            patch.data[:,2:,:]=0
            self.register_parameter("patch%d"%patchIndex,patch)
            self.grid.append(patch)
        #==============================================================================

    def forward(self, x):

        #encoder
        #==============================================================================
        x = self.encoder(x.transpose(2,1).contiguous())
        #==============================================================================

        outs = []
        patches = []

        for i in range(0,self.nb_patches):

            #random planar patch
            #==========================================================================
            rand_grid = self.grid[i].expand(x.size(0),-1,-1)
            patches.append(rand_grid[0].transpose(1,0))
            #==========================================================================

            #cat with latent vector and decode
            #==========================================================================
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
            #==========================================================================

        return torch.cat(outs,2).transpose(2,1).contiguous(), patches


class PointTransLinAdj(nn.Module):
    """Ours auto encoder"""

    def __init__(self, options):

        super(PointTransLinAdj, self).__init__()

        self.npoint = options.npoint
        self.nb_patches = options.nb_patches
        self.atlasnet_nlatent = options.atlasnet_nlatent
        self.patchDim = options.patchDim
        self.patchDeformDim = options.patchDeformDim
        self.nbatch = options.nbatch

        #encoder decoder and patch deformation module
        #==============================================================================
        self.encoder = PointNetfeat(self.npoint,self.atlasnet_nlatent)
        self.linearTransformMatrix = nn.ModuleList(linearAdj(dim=self.patchDim,atlasnet_nlatent=self.atlasnet_nlatent) for i in range(0,self.nb_patches))
        self.patchDeformation = nn.ModuleList(patchDeformationMLP(patchDim = self.patchDim, patchDeformDim = self.patchDeformDim) for i in range(0,self.nb_patches))
        #==============================================================================

        #patch
        #==============================================================================
        self.grid = []
        for patchIndex in range(self.nb_patches):
            patch = torch.nn.Parameter(torch.FloatTensor(1,self.patchDim,self.npoint//self.nb_patches))
            patch.data.uniform_(0,1)
            patch.data[:,2:,:]=0
            self.register_parameter("patch%d"%patchIndex,patch)
            self.grid.append(patch)
        #==============================================================================

    def forward(self, x):

        #encode data
        #==============================================================================
        x = self.encoder(x.transpose(2,1).contiguous())
        #==============================================================================

        outs = []
        patches = []
        for i in range(0,self.nb_patches):

            #random planar patch
            #==========================================================================
            rand_grid =self.grid[i].expand(x.size(0),-1,-1).transpose(2,1)
            patches.append(rand_grid[0])
            #==========================================================================

            #apply linear tranformation to the patch
            #==========================================================================
            R,T = self.linearTransformMatrix[i](x.unsqueeze(2))
            rand_grid = torch.bmm(rand_grid,R) + T
            outs.append(rand_grid)
            #==========================================================================

        return torch.cat(outs,1).contiguous().contiguous(), patches


class PatchDeformMLPAdj(nn.Module):
    """Atlas net auto encoder"""

    def __init__(self, options):

        super(PatchDeformMLPAdj, self).__init__()

        self.nb_patches = options.nb_patches
        self.atlasnet_nlatent = options.atlasnet_nlatent
        # self.nbatch = options.nbatch
        self.patchDim = 2
        self.patchDeformDim = 3
        self.trained_atlasnet_decoder = options.trained_atlasnet_decoder
        self.trained_atlasnet_patch_deformation = options.trained_atlasnet_patch_deformation

        #encoder decoder and patch deformation module
        #==============================================================================
        self.encoder = None
        self.decoder = nn.ModuleList([mlpAdj(atlasnet_nlatent = self.patchDeformDim + self.atlasnet_nlatent) for i in range(0,self.nb_patches)])
        if self.trained_atlasnet_decoder:
            if not os.path.isfile(self.trained_atlasnet_decoder):
                raise Exception('Not a valid path for AtlasNet decoder %s'%self.trained_atlasnet_decoder)
            self.decoder.load_state_dict(torch.load(self.trained_atlasnet_decoder))

        self.patchDeformation = nn.ModuleList(patchDeformationMLP(patchDim = self.patchDim, patchDeformDim = self.patchDeformDim) for i in range(0,self.nb_patches))
        if self.trained_atlasnet_patch_deformation:
            if not os.path.isfile(self.trained_atlasnet_patch_deformation):
                raise Exception('Not a valid path for AtlasNet patchdeformation %s'%self.trained_atlasnet_patch_deformation)
            self.patchDeformation.load_state_dict(torch.load(self.trained_atlasnet_patch_deformation))

        #==============================================================================
    @abc.abstractmethod
    def encode(self, x):
        pass
    def forward_inference_from_latent_space(self, x, input_grid):
        batch_size, _ = x.size()
        # To keep the same interface as before
        if len(input_grid.size()) < 4:
            input_grid = input_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #  To keep the same interface  as before
        outs = []
        patches = []
        for i in range(0,self.nb_patches):
            grid = self.patchDeformation[i](input_grid[:,i,:,:].contiguous())
            patches.append(grid[0].transpose(1,0))
            #==========================================================================
            #cat with latent vector and decode
            #==========================================================================
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
            #==========================================================================
        return torch.cat(outs,2).transpose(2,1).contiguous(), patches
    def forward(self, x):
        #encoder
        #==============================================================================
        x = self.encode(x)
        #==============================================================================
        #random planar patch
        #==========================================================================
        rand_grid = torch.FloatTensor(x.size(0),self.nb_patches,self.patchDim,self.npoint//self.nb_patches).cuda()
        rand_grid.data.uniform_(0,1)
        rand_grid[:,:,2:,:] = 0
        return self.forward_inference_from_latent_space(x, rand_grid)

class AE_PatchDeformMLPAdj(PatchDeformMLPAdj):
    def __init__(self, options):
        super(AE_PatchDeformMLPAdj, self).__init__(options)
        self.npoint = options.npoint
        self.encoder = PointNetfeat(self.npoint,self.atlasnet_nlatent)
        self.trained_atlasnet_encoder = options.trained_atlasnet_encoder
        if self.trained_atlasnet_encoder:
            if not os.path.isfile(self.trained_atlasnet_encoder):
                raise Exception('Not a valid path for AtlasNet encoder %s'%self.trained_atlasnet_encoder)
            self.encoder.load_state_dict(torch.load(self.trained_atlasnet_encoder))
    def encode(self, x):
        x = self.encoder(x.transpose(2,1).contiguous())
        return x

class SVR_PatchDeformMLPAdj(PatchDeformMLPAdj):
    def __init__(self, options):
        super(SVR_PatchDeformMLPAdj, self).__init__(options)
        self.encoder = torchvision.models.resnet18(num_classes=1024)
        self.trained_atlasnet_encoder = options.trained_atlasnet_encoder
        if self.trained_atlasnet_encoder:
            if not os.path.isfile(self.trained_atlasnet_encoder):
                raise Exception('Not a valid path for AtlasNet encoder %s'%self.trained_atlasnet_encoder)
            self.encoder.load_state_dict(torch.load(self.trained_atlasnet_encoder))
        for parameter in self.decoder.parameters():
            parameter.requires_grad = options.train_atlasnet_decoder

    def encode(self, x):
        x = self.encoder(x)
        return x

class PatchDeformLinAdj(nn.Module):
    """Ours auto encoder"""

    def __init__(self, options):

        super(PatchDeformLinAdj, self).__init__()

        self.npoint = options.npoint
        self.nb_patches = options.nb_patches
        self.atlasnet_nlatent = options.atlasnet_nlatent
        self.patchDim = options.patchDim
        self.patchDeformDim = options.patchDeformDim

        #encoder decoder and patch deformation module
        #==============================================================================
        self.encoder = PointNetfeat(self.npoint,self.atlasnet_nlatent)
        self.linearTransformMatrix = nn.ModuleList(linearAdj(dim = self.patchDeformDim,atlasnet_nlatent=self.atlasnet_nlatent) for i in range(0,self.nb_patches))
        self.patchDeformation = nn.ModuleList(patchDeformationMLP(patchDim = self.patchDim, patchDeformDim = self.patchDeformDim) for i in range(0,self.nb_patches))
        #==============================================================================

    def forward(self, x):

        #encode data
        #==============================================================================
        x = self.encoder(x.transpose(2,1).contiguous())
        #==============================================================================

        outs = []
        patches = []

        for i in range(0,self.nb_patches):

            #random planar patch
            #==========================================================================
            rand_grid = torch.FloatTensor(x.size(0),self.patchDim,self.npoint//self.nb_patches).cuda()
            rand_grid.data.uniform_(0,1)
            rand_grid[:,2:,:] = 0
            #==========================================================================

            #deform the planar patch
            #==========================================================================
            rand_grid = self.patchDeformation[i](rand_grid.contiguous()).transpose(2,1)
            patches.append(rand_grid[0])
            #==========================================================================

            #apply linear tranformation to the patch
            #==========================================================================
            R,T = self.linearTransformMatrix[i](x.unsqueeze(2))
            rand_grid = torch.bmm(rand_grid,R) + T
            outs.append(rand_grid)
            #==========================================================================

        return torch.cat(outs,1).contiguous().contiguous(), patches

class MODEL_LIST:
    """list of all the model"""
    def __init__(self, options):
        if options.adjust == 'mlp':
            if options.use_svr:
                self.models = {'AtlasNet':AtlasNet,'PointTranslation':PointTransMLPAdj,'PatchDeformation':SVR_PatchDeformMLPAdj}
            else:
                self.models = {'AtlasNet':AtlasNet,'PointTranslation':PointTransMLPAdj,'PatchDeformation':AE_PatchDeformMLPAdj}
        elif options.adjust == 'linear':
             self.models = {'AtlasNet':AtlasNetLinAdj,'PointTranslation':PointTransLinAdj,'PatchDeformation':PatchDeformLinAdj}
        else:
            print(colors.FAIL,"ERROR please select the model from : ",colors.ENDC)
            print("   > mlp")
            print("   > linear")
            exit()

        self.type = self.models.keys()

    def load(self,options):
        return self.models[options.model](options).cuda()
