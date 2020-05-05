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

evaluate = ['train', 'test']
for i in evaluate:
    _, r_hat, _ = carl.evaluate(x='data/'+do+'/x0_'+i+'.npy')
    w = 1./r_hat
    loading.load_result(x0='data/'+do+'/x0_'+i+'.npy',
                        x1='data/'+do+'/x1_train.npy',
                        weights=w, 
                        label = i+'_calibrated',
                        do = do,
    )
