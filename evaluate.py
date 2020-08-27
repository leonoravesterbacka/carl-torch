import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples',   action='store', type=str, dest='samples',   default='dilepton', help='samples to derive weights for. Sherpa 2.2.8 ttbar dilepton')
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='qsf', help='variation to derive weights for. default QSF down to QSF up')

(opts, args) = parser.parse_args()
do  = opts.samples
var = opts.variation

loading = Loader()
carl = RatioEstimator()
carl.load('models/'+ do + '/' + var + '_carl')
evaluate = ['train', 'val']
for i in evaluate:
    r_hat, _ = carl.evaluate(x='data/'+ do + '/' + var + '/X0_'+i+'.npy')
    w = 1./r_hat
    loading.load_result(x0='data/'+ do + '/' + var + '/X0_'+i+'.npy',     
                        x1='data/'+ do + '/' + var + '/X1_'+i+'.npy',
                        weights=w, 
                        label = i,
                        do = do,
                        var = var,
                        save = True,
    )
carl.evaluate_performance(x='data/'+ do + '/' + var + '/X_val.npy',y='data/' + do + '/' + var +'/y_val.npy')
