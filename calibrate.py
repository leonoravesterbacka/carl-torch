import numpy as np
import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader
from ml.calibration import CalibratedClassifier
from ml.base import Estimator

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='chose what samples to perform weight derivation on. default Sherpa     vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples


loading = Loader()

carl = RatioEstimator()
carl.load('models/'+do+'_carl')
#load
X = 'data/'+do+'/x_train.npy'
y = 'data/'+do+'/y_train.npy'
_, r_hat, _ = carl.evaluate(X)
print("uncalibrated weight is ",1/r_hat)
calib = CalibratedClassifier(carl)
calib.fit(X = X,y = y)
r = calib.predict(X ='data/'+do+'/x0_train.npy')
w = 1/r
print("calibrated weight r is ", w)
loading.load_result(x0='data/'+do+'/x0_train.npy',     
                    x1='data/'+do+'/x1_train.npy',
                    weights=w, 
                    label = 'calibrated',
                    do = do,
)              
