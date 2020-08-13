import logging
import optparse
from ml import RatioEstimator
from ml import Loader


parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='samples to derive weights for. default Sherpa vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples

loading = Loader()
loading.loading(
    folder='./data/',
    plot=True,
    do = do,
    randomize = False,
    save = True,
    correlation = True,
    preprocessing = True,
)
x='data/'+do+'/X_train.npy'
y='data/'+do+'/Y_train.npy'

estimator = RatioEstimator(
    n_hidden=(4,2),
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
estimator.save('models/'+do+'_carl', x=x, export_model = True)
