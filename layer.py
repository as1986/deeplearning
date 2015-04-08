from utils import *
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from learning_rule import *

rng = np.random.RandomState(42)
trng = T.shared_randomstreams.RandomStreams(42)

def prop(layers,x):
    for i, layer in enumerate(layers):
        if i == 0:
            layer_out = layer.fprop(x)
        else:
            layer_out = layer.fprop(layer_out)
    return layers[-1].h

class Projection:
    """
    Vector => Matrix Projcection Layer
    """
    def __init__(self, vocab_size=None, hid_dim=None, embedding=None, update=False):
        if embedding is None:
            self.vocab_size = vocab_size
            self.hid_dim = hid_dim
            self.embedding = create_embedding(self.vocab_size, self.hid_dim)
        else:
            self.embedding = create_shared(embedding)
        if update:
            self.params = [ self.embedding ]
        else:
            self.params = [ ]

    def fprop(self, x):
        h = self.embedding[x]
        self.h = h
        return h

class fProjection:
    """
    Matrix => Matrix Projcection Layer
    """
    def __init__(self, vocab_size=None, hid_dim=None, embedding=None, update=False):
        if embedding is None:
            self.vocab_size = vocab_size
            self.hid_dim = hid_dim
            self.embedding = create_embedding(self.vocab_size, self.hid_dim)
        else:
            self.embedding = create_shared(embedding)
        if update:
            self.params = [ self.embedding ]
        else:
            self.params = []

    def fprop(self, x):
        size = x.shape[0]
        x = x.flatten()
        tmp = self.embedding[x].flatten()
        h = tmp.reshape((size,tmp.shape[0]/size))
        self.h = h
        return h

class aProjection:
    """
    Matrix => Matrix Projcection Layer
    """
    def __init__(self, vocab_size=None, hid_dim=None, embedding=None, update=False):
        if embedding is None:
            self.vocab_size = vocab_size
            self.hid_dim = hid_dim
            self.embedding = create_embedding(self.vocab_size, self.hid_dim)
        else:
            self.embedding = create_shared(embedding)
        if update:
            self.params = [ self.embedding ]
        else:
            self.params = []

    def fprop(self, x):
        size = x.shape[0]
        h = self.embedding[x].mean(axis=1)
        self.h = h
        return h

class tProjection:
    """
    Matrix => Tensor Projection Layer
    """
    def __init__(self, vocab_size=None, hid_dim=None, embedding=None, orig=None):
        if orig is not None:
            self.copy_constructor(orig)
            return
        elif embedding is None:
            self.vocab_size = vocab_size
            self.hid_dim = hid_dim
            self.embedding = create_embedding(self.vocab_size, self.hid_dim)
        else:
            self.vocab_size = len(embedding)
            self.hid_dim = len(embedding[0])
            self.embedding = create_shared(embedding)
        self.params = [ self.embedding ]

    def fprop(self, x):
        h = self.embedding[x].dimshuffle((1,0,2))
        self.h = h
        return h

    def copy_constructor(self, orig):
        assert(isinstance(orig, tProjection))
        self.embedding = orig.embedding
        self.vocab_size = orig.vocab_size
        self.hid_dim = orig.hid_dim
        self.params = [ self.embedding ]

    def set_embeddings(self, embedding):
        self.vocab_size = len(embedding)
        self.hid_dim = len(embedding[0])
        self.embedding = create_shared(embedding)
        self.params = [ self.embedding ]


    def __getstate__(self):
        return (self.vocab_size, self.hid_dim, self.embedding.eval())

    def __setstate__(self, state):
        vocab_size, hid_dim, embedding = state
        self.vocab_size = len(embedding)
        self.hid_dim = len(embedding[0])
        self.embedding = create_shared(np.asarray(embedding))
        self.params = [ self.embedding ]
        return

class dProjection:
    """
    Dual Projcection Layer
    Example: http://www.aclweb.org/anthology/P14-1138.pdf
    """
    def __init__(self, F, E, update=True):
        self.F = create_shared(F) 
        self.E = create_shared(E) 
        if update:
            self.params = [ self.F, self.E ]
        else:
            self.params = []

    def fprop(self,f,e):
        h1 = self.F[f].flatten().reshape((f.shape[0]*f.shape[1],self.F.shape[1]))
        h2 = self.E[e].flatten().reshape((e.shape[0]*e.shape[1],self.E.shape[1]))
        h = T.concatenate([h1, h2], axis=1).flatten().reshape((f.shape[0],f.shape[1],self.F.shape[1]+self.E.shape[1])).dimshuffle(1,0,2)
        self.h = h
        return h

class Layer:
    def __init__(self, vis_dim=0, hid_dim=0, func=None, orig=None):
        if orig is not None:
            assert isinstance(orig, Layer)
            self.vis_dim = orig.vis_dim
            self.hid_dim = orig.hid_dim
            self.W = orig.W
            self.b = orig.b
            self.params = orig.params
            self.func = orig.func
            return
        self.vis_dim = vis_dim
        self.hid_dim = hid_dim
        self.W = create_weight(vis_dim,hid_dim)
        self.b = create_bias(hid_dim)
        self.params = [ self.W, self.b ]
        self.func = func

    def fprop(self,x):
        h = self.func(T.dot(x, self.W)+self.b)
        self.h = h
        return h

    def pretrain(self,X,epochs,batch_size=100,corruption_rate=0.0):
        X = create_shared(X)
        n_batches = X.get_value(borrow=True).shape[0] / batch_size

        x = T.fmatrix('x')
        W_prime = create_weight(self.hid_dim,self.vis_dim)
        b_prime = create_bias(self.vis_dim)
        x = x * trng.binomial(x.shape,p=1-corruption_rate, n=1, dtype=x.dtype)
        z = self.fprop(x)
        out = self.func(T.dot(z,W_prime) + b_prime)
        params = self.params + [ W_prime, b_prime ]
        cost = T.sum((x-out)**2)/2
        updates = learning_rule(cost, params, max_norm = 1.0, eps= 1e-6, rho=0.65, method = "adadelta")

        index = T.lscalar("index")
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        train = theano.function(
                inputs = [index], 
                outputs = cost, 
                updates = updates, 
                givens=[(x, X[batch_begin:batch_end])]
                )
        predict = theano.function([],z,givens=[(x,X)])
        for epoch in xrange(epochs):
            c = [train(batch_index) for batch_index in xrange(n_batches)]
            print("epoch: ", epoch, "COST: ", np.mean(c))
        return predict()


class EmbLayer:
    def __init__(self, vis_dim=0, hid_dim=0, func=None, orig=None):
        if orig is not None:
            self.copy_constructor(orig)
            return
        self.vis_dim = vis_dim
        self.hid_dim = hid_dim
        self.W = create_weight(vis_dim,hid_dim)
        self.b = create_bias(hid_dim)
        self.params = [ self.W, self.b ]
        self.func = func

    def copy_constructor(self, orig):
        assert isinstance(orig, EmbLayer)
        self.vis_dim = orig.vis_dim
        self.hid_dim = orig.hid_dim
        self.W = orig.W
        self.b = orig.b
        self.params = orig.params
        self.func = orig.func
        return
    def fprop(self,x):
        h = self.func(T.dot(x, self.W)+self.b)
        self.h = h
        return h

    def pretrain(self,X,epochs,batch_size=100,corruption_rate=0.0):
        X = create_shared(X)
        n_batches = X.get_value(borrow=True).shape[0] / batch_size

        x = T.fmatrix('x')
        W_prime = create_weight(self.hid_dim,self.vis_dim)
        b_prime = create_bias(self.vis_dim)
        x = x * trng.binomial(x.shape,p=1-corruption_rate, n=1, dtype=x.dtype)
        z = self.fprop(x)
        out = self.func(T.dot(z,W_prime) + b_prime)
        params = self.params + [ W_prime, b_prime ]
        cost = T.sum((x-out)**2)/2
        updates = learning_rule(cost, params, max_norm = 1.0, eps= 1e-6, rho=0.65, method = "adadelta")

        index = T.lscalar("index")
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        train = theano.function(
                inputs = [index],
                outputs = cost,
                updates = updates,
                givens=[(x, X[batch_begin:batch_end])]
                )
        predict = theano.function([],z,givens=[(x,X)])
        for epoch in xrange(epochs):
            c = [train(batch_index) for batch_index in xrange(n_batches)]
            print("epoch: ", epoch, "COST: ", np.mean(c))
        return predict()

class ConvLayer:
    """
    This layer is borrowed from http://deeplearning.net/tutorial/lenet.html
    """
    def __init__(self, filter_shape, image_shape, pool_size=(2,2)):
        """
            filter_shape = ( number of filters, num input feature maps, filter height, filter width )
            image_shape = ( batch size, feature maps=length, image height=context, image width=embedding_size )
            inputs =  feature maps * filter height * filter width
        """
        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))

        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = create_shared( 
                    np.asarray(
                        rng.uniform(
                            low= -1 * np.sqrt(6. / (fan_in + fan_out)),
                            high= np.sqrt(6. / (fan_in + fan_out)),
                            size=filter_shape
                        )
                    )
        )

        self.b = create_bias(filter_shape[0])
        self.params = [ self.W, self.b ]
    
    def fprop(self,x):
        conv_out = conv.conv2d(
                input=x,
                filters=self.W,
                filter_shape=self.filter_shape,
                image_shape=self.image_shape
                )

        pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=self.pool_size,
                ignore_border=True
                )
        
        h = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.h = h
        return h

class sConvLayer(ConvLayer):
    def __init__(self, filter_shape, image_shape, pool_size=(2,2)):
        ConvLayer.__init__(self, filter_shape, image_shape, pool_size)
        self.image_shape = image_shape

    def fprop(self,x):
        def step(u_t):
            u_t = u_t.reshape(self.image_shape)
            conv_out = conv.conv2d(
                    input=u_t,
                    filters=self.W,
                    filter_shape=self.filter_shape,
                    image_shape=self.image_shape
                    )

            pooled_out = downsample.max_pool_2d(
                    input=conv_out,
                    ds=self.pool_size,
                    ignore_border=True
                    )
            return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten()

        h, _ = theano.scan(fn = step, sequences = x)

        self.h = h
        return h

class RNN:
    def __init__(self, vis_dim, hid_dim, minibatch=True):
        prange = 4 * np.sqrt(6. / (vis_dim + hid_dim))
        self.hid_dim = hid_dim
        self.minibatch = minibatch
        
        self.Wx = create_shared(rng.randn(vis_dim, hid_dim) * prange)
        self.Wh = create_shared(rng.randn(hid_dim, hid_dim) * prange)
        self.bh = create_shared(rng.randn(hid_dim,) * prange)
        self.h0 = create_shared(np.zeros(hid_dim,))

        self.output_info = [ self.h0 ]
        self.params = [ self.Wx, self.Wh, self.bh ]

    def fprop(self,x):
        def step(u_t, h_tm1):
            h = T.nnet.sigmoid(T.dot(u_t,self.Wx) + T.dot(h_tm1,self.Wh) + self.bh)
            return h

        if self.minibatch:
            self.output_info = [ T.alloc(init, x.shape[1], self.hid_dim) for init in self.output_info ]

        h, _ = theano.scan(
            fn = step,
            sequences = x,
            outputs_info = self.output_info,
            n_steps=x.shape[0]
            )

        self.h = h
        return h

class LSTM:
    """
    Generating Sequence with Recurrent Neural Networks
    Alex Graves
    http://arxiv.org/pdf/1308.0850v5.pdf
    """
    def __init__(self, vis_dim=0, hid_dim=0, h0=None, minibatch=False, orig=None):
        if orig is not None:
            self.copy_constructor(orig)
            return
        prange = 1 * np.sqrt(6. / (vis_dim + hid_dim))
        self.hid_dim = hid_dim
        self.minibatch = minibatch

        # Weights
        self.W_u_ig = create_shared(rng.randn(vis_dim, hid_dim) * prange) 
        self.W_u_og = create_shared(rng.randn(vis_dim, hid_dim) * prange)
        self.W_u_fg = create_shared(rng.randn(vis_dim, hid_dim) * prange)
        self.W_u_in = create_shared(rng.randn(vis_dim, hid_dim) * prange)

        self.W_h_ig = create_shared(rng.randn(hid_dim, hid_dim) * prange) 
        self.W_h_og = create_shared(rng.randn(hid_dim, hid_dim) * prange) 
        self.W_h_fg = create_shared(rng.randn(hid_dim, hid_dim) * prange) 
        self.W_h_in = create_shared(rng.randn(hid_dim, hid_dim) * prange) 

        self.W_c_ig = create_shared(rng.randn(hid_dim, hid_dim) * prange) 
        self.W_c_og = create_shared(rng.randn(hid_dim, hid_dim) * prange) 
        self.W_c_fg = create_shared(rng.randn(hid_dim, hid_dim) * prange) 

        self.B_ig = create_shared(rng.randn(hid_dim,) * prange) 
        self.B_og = create_shared(rng.randn(hid_dim,) * prange) 
        self.B_fg = create_shared(rng.randn(hid_dim,) * prange) 
        self.B_in = create_shared(rng.randn(hid_dim,) * prange) 

        self.c0 = create_shared(np.zeros(hid_dim))
        if h0 is None:
            self.h0 = create_shared(np.zeros(hid_dim))
        else:
            self.h0 = h0

        # Params
        self.output_info = [self.c0, self.h0]
        self.params = [
            self.W_u_ig, self.W_u_og, self.W_u_fg, self.W_u_in, self.W_h_ig, self.W_h_og, self.W_h_fg, self.W_h_in,
            self.W_c_ig, self.W_c_og, self.W_c_fg, self.B_ig, self.B_og, self.B_fg, self.B_in
        ]

    def __getstate__(self):
        state = []
        state.append(self.hid_dim)
        state.append(self.minibatch)
        state.append(self.W_u_ig.eval())
        state.append(self.W_u_og.eval())
        state.append(self.W_u_fg.eval())
        state.append(self.W_u_in.eval())

        state.append(self.W_h_ig.eval())
        state.append(self.W_h_og.eval())
        state.append(self.W_h_fg.eval())
        state.append(self.W_h_in.eval())

        state.append(self.W_c_ig.eval())
        state.append(self.W_c_og.eval())
        state.append(self.W_c_fg.eval())

        state.append(self.B_ig.eval())
        state.append(self.B_og.eval())
        state.append(self.B_fg.eval())
        state.append(self.B_in.eval())

        state.append(self.c0.eval())
        state.append(self.h0.eval())

        return state

    def __setstate__(self, state):
        '''
        FIXME this is not satisfactory at all!!!
        :param state:
        :return:
        '''
        self.hid_dim = state[0]
        self.minibatch = state[1]
        self.W_u_ig = create_shared(np.asarray(state[2]))
        self.W_u_og = create_shared(np.asarray(state[3]))
        self.W_u_fg = create_shared(np.asarray(state[4]))
        self.W_u_in = create_shared(np.asarray(state[5]))

        self.W_h_ig = create_shared(np.asarray(state[6]))
        self.W_h_og = create_shared(np.asarray(state[7]))
        self.W_h_fg = create_shared(np.asarray(state[8]))
        self.W_h_in = create_shared(np.asarray(state[9]))

        self.W_c_ig = create_shared(np.asarray(state[10]))
        self.W_c_og = create_shared(np.asarray(state[11]))
        self.W_c_fg = create_shared(np.asarray(state[12]))

        self.B_ig = create_shared(np.asarray(state[13]))
        self.B_og = create_shared(np.asarray(state[14]))
        self.B_fg = create_shared(np.asarray(state[15]))
        self.B_in = create_shared(np.asarray(state[16]))

        self.c0 = create_shared(np.asarray(state[17]))
        self.h0 = create_shared(np.asarray(state[18]))

        self.output_info = [self.c0, self.h0]
        self.params = [
            self.W_u_ig, self.W_u_og, self.W_u_fg, self.W_u_in, self.W_h_ig, self.W_h_og, self.W_h_fg, self.W_h_in,
            self.W_c_ig, self.W_c_og, self.W_c_fg, self.B_ig, self.B_og, self.B_fg, self.B_in
        ]

        return

    def copy_constructor(self, orig):
        assert(isinstance(orig, LSTM))
        self.hid_dim = orig.hid_dim
        self.minibatch = orig.minibatch
        self.W_u_ig = orig.W_u_ig
        self.W_u_og = orig.W_u_og
        self.W_u_fg = orig.W_u_fg
        self.W_u_in = orig.W_u_in

        self.W_h_ig = orig.W_h_ig
        self.W_h_og = orig.W_h_og
        self.W_h_fg = orig.W_h_fg
        self.W_h_in = orig.W_h_in

        self.W_c_ig = orig.W_c_ig
        self.W_c_og = orig.W_c_og
        self.W_c_fg = orig.W_c_fg

        self.B_ig = orig.B_ig
        self.B_og = orig.B_og
        self.B_fg = orig.B_fg
        self.B_in = orig.B_in

        self.c0 = orig.c0
        self.h0 = orig.h0
        self.output_info = [self.c0, self.h0]
        self.params = [
            self.W_u_ig, self.W_u_og, self.W_u_fg, self.W_u_in, self.W_h_ig, self.W_h_og, self.W_h_fg, self.W_h_in,
            self.W_c_ig, self.W_c_og, self.W_c_fg, self.B_ig, self.B_og, self.B_fg, self.B_in
        ]
        return
    
    def fprop(self,x):
        """
            x : n_step * batch_size * n_hid
        """
        def step(u_t, c_tm1, h_tm1):
            # Gates 
            ig = T.nnet.sigmoid(T.dot(u_t, self.W_u_ig) + T.dot(h_tm1, self.W_h_ig) + T.dot(c_tm1, self.W_c_ig) + self.B_ig)
            og = T.nnet.sigmoid(T.dot(u_t, self.W_u_og) + T.dot(h_tm1, self.W_h_og) + T.dot(c_tm1, self.W_c_og) + self.B_og)
            fg = T.nnet.sigmoid(T.dot(u_t, self.W_u_fg) + T.dot(h_tm1, self.W_h_fg) + T.dot(c_tm1, self.W_c_fg) + self.B_fg)
            # Cell input 
            ci = T.tanh(T.dot(u_t, self.W_u_in) + T.dot(h_tm1, self.W_h_in) + self.B_in)
            c_t = ci* ig + fg * c_tm1
            h_t = T.tanh(c_t) * og
            return c_t, h_t

        if self.minibatch:
            self.output_info = [ T.alloc(init, x.shape[1], self.hid_dim) for init in self.output_info ]

        [_, h], _ = theano.scan(
            fn = step,
            sequences = x,
            outputs_info = self.output_info, 
            n_steps = x.shape[0]
            )

        self.h = h
        return h

class cLSTM(LSTM):
    """
    Classification LSTM, this layer can include output layer ( linear, sigmoid, softmax .. etc ) 
    """
    def __init__(self, vis_dim, hid_dim, out_dim, func, h0=None, minibatch=True):
        LSTM.__init__(self, vis_dim, hid_dim, h0=h0, minibatch=minibatch)
        self.out_dim = out_dim
        self.W = create_weight(self.hid_dim,out_dim)
        self.b = create_bias(out_dim)
        self.params = self.params + [ self.W, self.b ]
        self.func = func

    def fprop(self,x):
        """
            x : n_step * batch_size * n_hid
        """
        def step(u_t, c_tm1, h_tm1):
            # Gates 
            ig = T.nnet.sigmoid(T.dot(u_t, self.W_u_ig) + T.dot(h_tm1, self.W_h_ig) + T.dot(c_tm1, self.W_c_ig) + self.B_ig)
            og = T.nnet.sigmoid(T.dot(u_t, self.W_u_og) + T.dot(h_tm1, self.W_h_og) + T.dot(c_tm1, self.W_c_og) + self.B_og)
            fg = T.nnet.sigmoid(T.dot(u_t, self.W_u_fg) + T.dot(h_tm1, self.W_h_fg) + T.dot(c_tm1, self.W_c_fg) + self.B_fg)
            # Cell input 
            c_inp = T.tanh(T.dot(u_t, self.W_u_in) + T.dot(h_tm1, self.W_h_in) + self.B_in)
            c_t = c_inp * ig + fg * c_tm1
            h_t = T.tanh(c_t) * og
            o_t = self.func(T.dot(h_t, self.W) + self.b)
            return c_t, h_t, o_t

        if self.minibatch:
            self.output_info = [ T.alloc(init, x.shape[1], self.hid_dim) for init in self.output_info ]
        self.output_info += [None]

        [_, _, h], _ = theano.scan(
            fn = step,
            sequences = x,
            outputs_info = self.output_info, 
            n_steps = x.shape[0]
            )

        self.h = h
        return h

class dLSTM(LSTM):
    def __init__(self, vis_dim, hid_dim, out_dim, func, h0=None, minibatch=True):
        LSTM.__init__(self, vis_dim, hid_dim, h0=h0, minibatch=minibatch)
        self.out_dim = out_dim
        self.W = create_weight(self.hid_dim,out_dim)
        self.b = create_bias(out_dim)
        self.output_info = [self.c0, self.h0]
        self.params = self.params + [ self.W, self.b ]
        self.func = func

    def fprop(self,x):
        def step(u_t, c_tm1, h_tm1, o_tm1):
            # Gates 
            ig = T.nnet.sigmoid(T.dot(o_tm1, self.W_u_ig) + T.dot(h_tm1, self.W_h_ig) + T.dot(c_tm1, self.W_c_ig) + self.B_ig)
            og = T.nnet.sigmoid(T.dot(o_tm1, self.W_u_og) + T.dot(h_tm1, self.W_h_og) + T.dot(c_tm1, self.W_c_og) + self.B_og)
            fg = T.nnet.sigmoid(T.dot(o_tm1, self.W_u_fg) + T.dot(h_tm1, self.W_h_fg) + T.dot(c_tm1, self.W_c_fg) + self.B_fg)
            # Cell input 
            c_inp = T.tanh(T.dot(o_tm1, self.W_u_in) + T.dot(h_tm1, self.W_h_in) + self.B_in)
            c_t = c_inp * ig + fg * c_tm1
            h_t = T.tanh(c_t) * og
            o_t = self.func(T.dot(h_t, self.W)+self.b)
            return c_t, h_t, o_t

        if self.minibatch:
            self.output_info = [ T.alloc(init, x.shape[1], self.hid_dim) for init in self.output_info ]
        self.output_info += [ x[0] ]

        [_, _, h], _ = theano.scan(
            fn = step,
            sequences = x,
            outputs_info = self.output_info,
            n_steps = 1024
            )

        self.h = h
        return h
