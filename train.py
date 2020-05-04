import logging
import optparse
from ml import RatioEstimator
from ml import Loader

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='chose what samples to perform weight derivation on. default Sherpa vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples

loading = Loader()
x, y = loading.loading(
    folder='./data/',
    filename='train',
    plot=True,
    do = do,
)
estimator = RatioEstimator(
    n_hidden=(10,10),
    activation="sigmoid"
)
estimator.train(
    method='carl',
    batch_size = 128,
    n_epochs = 50,
    x='data/'+do+'/x_train.npy',
    y='data/'+do+'/y_train.npy',
    scale_inputs = True,
)
estimator.save('models/'+do+'_carl')
