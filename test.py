from layer import *
from learning_rule import * 
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support

def mnist_test(batch_size=10):
    mnist = fetch_mldata('MNIST original')
    X = mnist.data/float(255)
    Y = mnist.target
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

    layers = [
        Layer(784,1000,T.nnet.sigmoid),
        Layer(1000,1000,T.nnet.sigmoid),
        Layer(1000,1000,T.nnet.sigmoid),
        Layer(1000,10,T.nnet.softmax),
    ]

    inp = train_x
    for layer in layers[:-1]:
        inp = layer.pretrain(inp,epochs=100,corruption_rate=0.3)

    params = []
    for layer in layers:
        params += layer.params

    x = T.fmatrix("x")
    for i, layer in enumerate(layers):
        if i == 0:
            layer_out = layer.fprop(x)
        else:
            #layer_out = layer_out * trng.binomial(layer_out.shape,p=0.5, n=1, dtype=layer_out.dtype)
            layer_out = layer.fprop(layer_out)
    t = T.matrix("t")
    y = layers[-1].h
    cost = T.nnet.binary_crossentropy(y, t).mean() 
    updates = learning_rule(cost, params, max_norm = 5.0, eps= 1e-6, rho=0.65, method = "rmsprop")

    train_X = create_shared(train_x.astype(theano.config.floatX))
    train_Y = create_shared(preprocessing.LabelBinarizer().fit_transform(train_y).astype(theano.config.floatX))
    test_X = create_shared(test_x.astype(theano.config.floatX))
    n_batches = train_X.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar("index")
    batch_begin = index * batch_size
    batch_end = batch_begin + batch_size
    train = theano.function(
            inputs = [index], 
            outputs = cost, 
            updates = updates, 
            givens=[(x, train_X[batch_begin:batch_end]),(t, train_Y[batch_begin:batch_end])]
            )
    predict = theano.function([],y,givens=[(x,test_X)])
    for _ in range(1000):
        cost = np.mean([ train(i) for i in range(n_batches)])
        pred_y = np.argmax(predict(),axis=1)
        print precision_recall_fscore_support(test_y, pred_y, average='macro') 


def cmnist_test(batch_size=10):
    mnist = fetch_mldata('MNIST original')
    X = mnist.data/float(255)
    Y = mnist.target
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_X = create_shared(train_x.astype(theano.config.floatX))
    train_Y = theano.shared(train_y.astype("int32"))
    test_X = create_shared(test_x.astype(theano.config.floatX))
    n_batches = train_X.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_X.get_value(borrow=True).shape[0] / batch_size

    x = T.matrix('x') 
    x_t = x.reshape((batch_size, 1, 28, 28))
    t = T.ivector('t')

    clayers = [
                ConvLayer(filter_shape=(20, 1, 5, 5),image_shape=(batch_size, 1, 28, 28),pool_size=(2, 2)),
                ConvLayer(filter_shape=(10, 20, 5, 5),image_shape=(batch_size, 20, 12, 12),pool_size=(2, 2)),
            ]

    layers = [
                Layer(vis_dim=10 * 4 * 4, hid_dim=500, func= T.nnet.sigmoid),
                Layer(vis_dim=500, hid_dim=500, func= T.nnet.sigmoid),
                Layer(vis_dim=500, hid_dim=500, func= T.nnet.sigmoid),
                Layer(vis_dim=500, hid_dim=10, func= T.nnet.softmax)
            ]

    params = []
    for layer in clayers + layers:
        params += layer.params

    for i, layer in enumerate(clayers):
        if i == 0:
            layer_out = layer.fprop(x_t)
        else:
            layer_out = layer.fprop(layer_out)

    for i, layer in enumerate(layers):
        if i == 0:
            layer_out = layer.fprop(layer_out.flatten(2))
        else:
            layer_out = layer.fprop(layer_out)

    y = layers[-1].h
    cost = - T.mean((T.log(y))[T.arange(x.shape[0]), t])
    updates = learning_rule(cost, params, max_norm = 1.0, eps= 1e-6, rho=0.95, method = "adadelta")

    print "Done"
    index = T.iscalar("index")
    batch_begin = index * batch_size
    batch_end = batch_begin + batch_size
    train = theano.function(
            inputs = [index], 
            outputs = cost, 
            updates = updates, 
            givens=[(x, train_X[batch_begin:batch_end]),(t, train_Y[batch_begin:batch_end])]
            )

    predict = theano.function(
            inputs = [index], 
            outputs = y, 
            givens=[(x, test_X[batch_begin:batch_end])]
            )

    for _ in range(10000):
        cost = np.mean([ train(i) for i in range(n_batches)])
        pred_y = []
        for j in range(n_test_batches):
            pred_y += list(np.argmax(predict(j),axis=1))
        print precision_recall_fscore_support(test_y, pred_y, average='macro') 


def minibatch_rnn_test():
    n_word = 100
    e_dim = 100
    n_class = 10
    n_context = 5
    batch_size = 2
    x = T.imatrix('x')
    t = T.imatrix('t')
    layers = [
        tProjection(n_word, e_dim),
        cLSTM(e_dim,200,n_class,T.nnet.softmax, minibatch=True)
    ]

    t_layer = tProjection(embedding=np.eye(n_class))
    label = t_layer.fprop(t)

    params = []
    for layer in layers:
        params += layer.params
    
    for i, layer in enumerate(layers):
        if i == 0:
            layer_out = layer.fprop(x)
        else:
            layer_out = layer.fprop(layer_out)

    y = layers[-1].h
    p = T.argmax(y, axis=2).T
    cost = -1 * (T.log(y)*label).sum() / t.sum()

    updates = learning_rule(cost, params, max_norm = 1.0, eps = 1e-6, rho = 0.65, clip = 1.0, method = "adadelta")

    train = theano.function([x,t],[cost,y,p],updates=updates)
    predict = theano.function([x],p)
    for _ in range(100):
        c,y_t,p_t = train(np.array([[1,2,3,0,0],[6,2,3,4,5]]).astype("int32"), np.array([[1,2,3,0,0],[3,3,3,3,3]]).astype("int32"))

        print predict(np.array([[1,2,3,0,0],[6,2,3,4,5]]).astype("int32"))
        print c

if __name__ == "__main__":
    cmnist_test()
