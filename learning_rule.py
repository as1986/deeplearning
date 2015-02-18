from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T

def learning_rule(cost, params, lr = 0.01, momentum = 0.9, eps= 1e-6, decay=0.1, rho=0.95, method = "adadelta"):
    lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
    momentum = theano.shared(np.float64(momentum).astype(theano.config.floatX))
    decay = np.float64(decay).astype(theano.config.floatX)
    eps = np.float64(eps).astype(theano.config.floatX)
    rho = theano.shared(np.float64(rho).astype(theano.config.floatX))

    gmomentums = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX)) if method == 'sgd' else None for param in params]
    gmss = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX)) if method == 'rmsprop' else None for param in params]
    gsums = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX)) if (method == 'adadelta' or method == 'adagrad') else None for param in params]
    xsums = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX)) if method == 'adadelta' else None for param in params]

    gparams = T.grad(cost, params)
    updates = OrderedDict()

    for gparam, param, gmomentum, gms, gsum, xsum in zip(gparams, params, gmomentums, gmss, gsums, xsums):
        
        gparam = T.switch(gparam.norm(L=2) > 5, 5*gparam/gparam.norm(L=2), gparam) 

        if method == 'adadelta':
            updates[gsum] = T.cast(rho * gsum + (1. - rho) * (gparam **2), theano.config.floatX)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] = T.cast(rho * xsum + (1. - rho) * (dparam **2), theano.config.floatX)
            updates[param] = T.cast(param + dparam, theano.config.floatX)
        elif method == 'adagrad':
            updates[gsum] =  T.cast(gsum + (gparam ** 2), theano.config.floatX)
            updates[param] =  T.cast(param - lr * (gparam / (T.sqrt(updates[gsum] + eps))), theano.config.floatX)
        elif method == 'rmsprop':
            updates[gms] = T.cast(decay * gms + (1 - decay) * T.sqr(gparam), theano.config.floatX)
            grms = T.cast(T.maximum(T.sqrt(updates[gms]), eps), theano.config.floatX)
            updates[param] = T.cast(param - lr * gparam / grms, theano.config.floatX)
        elif method == 'sgd':
            updates[gmomentum] = T.cast(momentum * gmomentum - lr * gparam, theano.config.floatX)
            updates[param] = T.cast(param + updates[gmomentum], theano.config.floatX)
        else:
            raise NotImplementedError
    
    return updates
