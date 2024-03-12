

# -*- coding: utf-8 -*-

"""


Created on Jul 21 12:21:36 2022
@author: HUI_ZHANG
E-mail : 202234949@mail.sdu.edu.cn  OR  bigserendipty@gmail.com

This implementation is the latest version of MBFAD-EAO algorithm.
"""


import time
import torch
import functorch as fc
from functools import partial
from loss import functional_xent, _xent


def MBFAD_EAO(model, dataset, config):
    
    config.lr = config.lr
    params    = tuple(model.parameters())
    train_loss = []
    test_loss  = []
    time_loss  = []
    train_x = dataset["train feature"]
    train_y = dataset["train label"]
    test_x  = dataset["test feature"]
    test_y  = dataset["test label"]
    
    with torch.no_grad(): # No Auto Diff
        
        model.train()
        fmodel, params = fc.make_functional(model)

        R = []
        C = []
        r = []
        c = []
        N_ones = []
        M_ones = []
        M = []
        
        i = 0
        for p in params:
            if i<=2:
                R.append(torch.unsqueeze(torch.zeros_like(p[:,0]), dim=1))
                C.append(torch.unsqueeze(torch.zeros_like(p[0,:]), dim=1).T)
                r.append(torch.unsqueeze(torch.zeros_like(p[:,0]), dim=1))
                c.append(torch.unsqueeze(torch.zeros_like(p[0,:]), dim=1).T)
                N_ones.append(torch.unsqueeze(torch.ones_like(p[:,0]), dim=1))
                M_ones.append(torch.unsqueeze(torch.ones_like(p[0,:]), dim=1))
                M.append(torch.zeros_like(p))
            else:
                R.append(torch.unsqueeze(torch.zeros(p.shape[0]), dim=1))
                C.append(torch.unsqueeze(torch.zeros(1), dim=1).T)
                r.append(torch.unsqueeze(torch.zeros(p.shape[0]), dim=1))
                c.append(torch.unsqueeze(torch.zeros(1), dim=1).T)
                N_ones.append(torch.unsqueeze(torch.ones(p.shape[0]), dim=1))
                M_ones.append(torch.unsqueeze(torch.ones(1), dim=1))
                M.append(torch.unsqueeze(torch.zeros(p.shape[0]), dim=1))
            i = i  + 1

        for epoch in range(config.epochs):
            
            # train
            t0 = time.perf_counter()
            batch_data = torch.randint(low=0, high = train_x.shape[0], size = (config.batch_size, ), generator = None, out = None, dtype = None, layout = torch.strided, device = None, requires_grad = False)            
            v_params = tuple([torch.randn_like(p) for p in params])
            " forword "
            f = partial(
                        functional_xent,
                        model = fmodel,
                        x = train_x[batch_data,:],
                        t = train_y[batch_data],
            )
            " jvp "
            loss_train, jvp = fc.jvp(f, (tuple(params),), (v_params,))
            
            " update parameters"
            for j, p in enumerate(params):
                
                # forward gradient
                grad = jvp * v_params[j]
                
                if j <=2:
                    r[j] = config.beta2 * r[j] + (1 - config.beta2) * torch.mm((torch.mul(grad, grad) + config.eta1 * torch.mm(N_ones[j], M_ones[j].T)), M_ones[j])
                    c[j] = config.beta2 * c[j] + (1 - config.beta2) * torch.mm(N_ones[j].T, (torch.mul(grad, grad) + config.eta1 * torch.mm(N_ones[j], M_ones[j].T)))
                    v = torch.mm(r[j], c[j])/torch.mm(N_ones[j].T, r[j])
                    u = grad/v.sqrt()
                    RMS = ((u-torch.mean(u))**2).sqrt()/config.d
                    u_bar = u/torch.max(torch.ones_like(u), RMS)
                    M[j] = config.beta1 * M[j] + (1-config.beta1)*u_bar
                    U = torch.mul(u_bar-M[j], u_bar-M[j])
                    R[j] = config.beta3*R[j] + (1-config.beta3) * torch.mm((U + config.eta2 * torch.mm(N_ones[j], M_ones[j].T)), M_ones[j])
                    C[j] = config.beta3*C[j] + (1-config.beta3) * torch.mm(N_ones[j].T, (U + config.eta2 * torch.mm(N_ones[j], M_ones[j].T)))
                    S = torch.mm(R[j], C[j])/torch.mm(N_ones[j].T, R[j])
                    p.sub_(torch.mul((config.lr/S.sqrt()), M[j]))

                else:
                    r[j] = config.beta2 * r[j] + (1 - config.beta2) * torch.mm((torch.mul(torch.unsqueeze(grad, dim=1), torch.unsqueeze(grad, dim=1)) + config.eta1 * torch.mm(N_ones[j], M_ones[j].T)), M_ones[j])
                    c[j] = config.beta2 * c[j] + (1 - config.beta2) * torch.mm(N_ones[j].T, (torch.mul(torch.unsqueeze(grad, dim=1), torch.unsqueeze(grad, dim=1)) + config.eta1 * torch.mm(N_ones[j], M_ones[j].T)))
                    v = torch.mm(r[j], c[j])/torch.mm(N_ones[j].T, r[j])
                    u = torch.unsqueeze(grad, dim=1)/v.sqrt()
                    
                    RMS = ((u-torch.mean(u))**2).sqrt()/config.d
                    u_bar = u/torch.max(torch.ones_like(u), RMS)
                    M[j] = config.beta1 * M[j] + (1-config.beta1)*u_bar
                    U = torch.mul(u_bar-M[j], u_bar-M[j])
                    R[j] = config.beta3*R[j] + (1-config.beta3) * torch.mm((U + config.eta2 * torch.mm(N_ones[j], M_ones[j].T)), M_ones[j])
                    C[j] = config.beta3*C[j] + (1-config.beta3) * torch.mm(N_ones[j].T, (U + config.eta2 * torch.mm(N_ones[j], M_ones[j].T)))
                    S = torch.mm(R[j], C[j])/torch.mm(N_ones[j].T, R[j])
                    update = torch.mul((config.lr/S.sqrt()), M[j])
                    p.sub_(update.reshape(update.shape[1], -1)[0,:])


            t1 = time.perf_counter()
            
            # update model parameters
            model.center.data[:,:] = params[0]
            model.sigma.data[:,:]  = params[1]
            model.ConsequentLayer.weight[:,:] = params[2]
            model.ConsequentLayer.bias[:] = params[3]

            # test
            y = model(test_x, droprule = False)
            loss_test = _xent(y, test_y)
            
            print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {loss_train:.4f}, Test Loss: {loss_test:.4f}, Time Loss(s): {t1 - t0:.4f}")

            # OK
            train_loss.append(loss_train.item())
            time_loss.append(t1 - t0)
            test_loss.append(loss_test.item())

    return train_loss, test_loss, time_loss

