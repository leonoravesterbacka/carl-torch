import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader
from ml.calibration import CalibratedClassifier
from ml.base import Estimator

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='samples to derive weights for. default Sherpa vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples

loading = Loader()

carl = RatioEstimator()
carl.load('models/'+do+'_carl')
#load
X  = 'data/'+do+'/x_train.npy'
y  = 'data/'+do+'/y_train.npy'
X0 = 'data/'+do+'/x0_train.npy'
X1 = 'data/'+do+'/x1_train.npy'
s_hat, r_hat, _ = carl.evaluate(X)
calib = CalibratedClassifier(carl)
calib.fit(X = X,y = y)
p0, p1, r_cal = calib.predict(X = X)
w_cal = 1/r_cal
loading.load_calibration(y_true = y,
                         p1_raw = s_hat, 
                         p1_cal = p1, 
                         label = 'calibrated',
                         do = do,
)

_, _, r_cal = calib.predict(X = X0)
w_cal = 1/r_cal
loading.load_result(x0 = X0,
                    x1 = X1,
                    weights=w_cal,
                    label = 'calibration_train',
                    do = do,
)              
