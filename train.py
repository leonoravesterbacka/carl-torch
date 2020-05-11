import logging
import optparse
from ml import RatioEstimator
from ml import Loader

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='samples to derive weights for. default Sherpa vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples

loading = Loader()
x, y = loading.loading(
    folder='./data/',
    plot=True,
    do = do,
    save = True,
)
estimator = RatioEstimator(
    n_hidden=(10,10),
    activation="relu"
)
estimator.train(
    method='carl',
    batch_size = 128,
    n_epochs = 10,
    x='data/'+do+'/x_train.npy',
    y='data/'+do+'/y_train.npy',
    scale_inputs = True,
)
estimator.save('models/'+do+'_carl', x='data/'+do+'/x_train.npy', export_model = True)
