import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader
from ml.calibration import CalibratedClassifier
from ml.base import Estimator

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples',   action='store', type=str, dest='samples',   default='dilepton', help='samples to derive weights for. Sherpa 2.2.8 ttbar dilepton')
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='qsf', help='variation to derive weights for. default QSF down to QSF up')
(opts, args) = parser.parse_args()
do  = opts.samples
var = opts.variation

loading = Loader()

carl = RatioEstimator()
carl.load('models/'+ do + '/' + var + '_carl')
#load
evaluate = ['train']
X  = 'data/'+ do + '/' + var + '/X_train.npy'
y  = 'data/'+ do + '/' + var + '/y_train.npy'
r_hat, s_hat = carl.evaluate(X)
calib = CalibratedClassifier(carl)
calib.fit(X = X,y = y)
p0, p1, r_cal = calib.predict(X = X)
w_cal = 1/r_cal
loading.load_calibration(y_true = y,
                         p1_raw = s_hat, 
                         p1_cal = p1, 
                         label = 'calibrated',
                         do = do,
                         var = var,
                         save = True,
)

evaluate = ['train']
for i in evaluate:
    p0, p1, r_cal = calib.predict(X = 'data/'+ do + '/' + var + '/X0_'+i+'.npy')
    w = 1./r_cal
    loading.load_result(x0='data/'+ do + '/' + var + '/X0_'+i+'.npy',
                        x1='data/'+ do + '/' + var + '/X1_'+i+'.npy',
                        weights=w, 
                        label = i+'_calib',
                        do = do,
                        save = True,
    )

