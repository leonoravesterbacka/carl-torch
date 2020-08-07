import logging
import optparse
from ml import RatioEstimator
from ml import Loader


parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='samples to derive weights for. default Sherpa vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples

#either train weight derivation model or classifier (the latter is used for hyperparameter search, validation etc.)
doWeights = True #if False, train model for hyperparameter search
loading = Loader()
loading.loading(
    folder='./data/',
    plot=True,
    do = do,
    randomize = True,
    save = True,
    correlation = False,
)
if doWeights:
    x='data/'+do+'/x_train.npy'
    y='data/'+do+'/y_train.npy'
    z=''
else:
    x='data/'+do+'/X_train.npy'
    y='data/'+do+'/Y_train.npy'
    z='val'

estimator = RatioEstimator(
    n_hidden=(10,10),
    activation="relu"
)
estimator.train(
    method='carl',
    batch_size = 128,
    n_epochs = 50,
    x=x,
    y=y,
    scale_inputs = True,
)
estimator.save('models/'+do+'_carl'+z, x=x, export_model = True)
