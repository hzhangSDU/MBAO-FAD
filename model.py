
# -*- coding: utf-8 -*-

"""


Created on Jul 20 20:35:24 2022
@author: HUI_ZHANG
E-mail : 202234949@mail.sdu.edu.cn  OR  bigserendipty@gmail.com

This implementation is the latest version of Takagi-Sugeno-Kang Fuzzy System(TSK-FS) class.
The implementation of DropRule technology refers to https://github.com/drwuHUST/MBGD_RDA.
"""

import torch
from torch import nn
from utils import IDX2VEC
torch.set_default_tensor_type(torch.DoubleTensor)



class TSK_FS(nn.Module):
    
    def __init__(self, M: int, nMFs: int, P: float):
        super(TSK_FS, self).__init__()
        
        self.M = M  # the dimension of input.
        self.nMFs = nMFs  # the number of Gaussian MFs in each input domain.
        self.R =  nMFs ** M  # the number of rules in the TSK fuzzy system.
        self.P = P  # the DropRule rate in MBFAD-EAO algorithm.
        
        self.__build_Antecedent__()
        self.__build_Consequent__()
        self.init_Consequent()

        
    def __build_Antecedent__(self):
        self.center = nn.Parameter(torch.rand(size=( self.M, self.nMFs )))  # center of the Gaussian MF.
        self.sigma  = nn.Parameter(torch.rand(size=( self.M, self.nMFs )))  # standard deviation of the Gaussian MF.

    def __build_Consequent__(self):
        self.ConsequentLayer = nn.Linear(self.M, self.R)  # fuzzy rules.
        
        
    def init_Antecedent(self, x: torch.Tensor):
        for m in range(self.M):
            with torch.no_grad():
                self.center[m,:] = torch.linspace(start=x[:,m].min(), end=x[:,m].max(), steps=self.nMFs, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    
    def init_Consequent(self):
        with torch.no_grad():
            a = torch.rand(size=(self.R, (self.M + 1)))
            self.ConsequentLayer.weight[:,:] = a[:,0:self.M]
            self.ConsequentLayer.bias[:] = a[:,self.M]


    def forward(self, x: torch.Tensor, droprule = True) -> torch.Tensor:

        MembershipGrade  = torch.zeros(size=( x.shape[0], self.M, self.nMFs ))
        FiringGrade      = torch.ones(size=( x.shape[0], self.R))
        FiringGrade_Bar  = torch.ones(size=( x.shape[0], self.R))
        nMFsVec          = self.nMFs * torch.ones(size=(self.M, 1))
        
        for n in range(x.shape[0]):
        
            # compute membership grade.
            MembershipGrade[n] = torch.exp(-(torch.sub(x[n],self.center.T))**2/2*self.sigma.T**2.0).T
            
            # drop rule.
            if droprule is True:
                FiringGrade[n, :] = self.__drop_rule(FiringGrade[n, :])
                
            # compute firing grade.
            for r in range(self.R):
                idsMFs = IDX2VEC(r+1, nMFsVec, self.nMFs, self.M)
                for m in range(self.M):
                    FiringGrade[n, r] = FiringGrade[n, r] * MembershipGrade[n, m, idsMFs[m]-1]
     
            FiringGrade_Bar[n,:] = FiringGrade[n,:]/torch.sum(FiringGrade[n,:])

        # compute rule output.
        Rule_Output = getattr(self, f"ConsequentLayer")(x)
        
        # compute output of TSK-FS.
        Output  = torch.sum(FiringGrade_Bar * Rule_Output, dim=1, out=None)

        return Output



    # the implementation of drop-rule
    def __drop_rule(self, FiringGrade):
        
        idsKeep = torch.rand(self.R)
        for i in range(self.R):
            
            if idsKeep[i] <= self.P:
                FiringGrade[i] = 1
            else:
                FiringGrade[i] = 0
        
        return FiringGrade

