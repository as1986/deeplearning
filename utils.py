import numpy as np
import theano
import theano.tensor as T

rng = np.random.RandomState(1234)
trng = T.shared_randomstreams.RandomStreams(1234)

def create_shared(X):
    return theano.shared(X.astype(theano.config.floatX))

def create_embedding(vocab_size,hid_dim,embedding=None):
    if embedding is not None:
        return create_shared(embedding.astype(theano.config.floatX))
    else:
        return create_shared(np.asarray(
                    np.r_[np.zeros((1,hid_dim)), rng.uniform(low=-1,high=1,size=(vocab_size,hid_dim))]
               ))

def create_weight(vis_dim,hid_dim):
    return create_shared(np.asarray(
                rng.uniform(
                    low=-4 * np.sqrt(6. / (vis_dim + hid_dim)),
                    high=4 * np.sqrt(6. / (vis_dim + hid_dim)),
                    size=(vis_dim,hid_dim)
                )
           ))

def create_bias(dim):
    return create_shared(np.zeros((dim,)))
