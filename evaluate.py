import os
import sys
import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples',   action='store', type=str, dest='samples',   default='dilepton', help='samples to derive weights for. Sherpa 2.2.8 ttbar dilepton')
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='qsf', help='variation to derive weights for. default QSF down to QSF up')
parser.add_option('-n', '--nentries',  action='store', type=str, dest='nentries',  default=0, help='specify the number of events do do the training on, default None means full sample')

(opts, args) = parser.parse_args()
sample = opts.samples
var    = opts.variation
n      = opts.nentries

if os.path.exists('data/'+ sample +'/'+ var +'/X_train_'+str(n)+'.npy'):
    print("Doing evaluation of model trained with datasets: ",sample, ", generator variation: ", var, " with ", n, " events." )
else:
    print("No datasets available for evaluation of model trained with ",sample, ", generator variation: ", var, " with ", n, " events." )
    print("ABORTING")
    sys.exit()
    
loading = Loader()
carl = RatioEstimator()
carl.load('models/'+ sample + '/' + var + '_carl_'+str(n))
evaluate = ['train', 'val']
for i in evaluate:
    r_hat, _ = carl.evaluate(x='data/'+ sample + '/' + var + '/X0_'+i+'_'+str(n)+'.npy')
    w = 1./r_hat
    loading.load_result(x0='data/'+ sample + '/' + var + '/X0_'+i+'_'+str(n)+'.npy',     
                        x1='data/'+ sample + '/' + var + '/X1_'+i+'_'+str(n)+'.npy',
                        weights=w, 
                        label = i,
                        do = sample,
                        var = var,
                        save = True,
                        n = n,
    )
carl.evaluate_performance(x='data/'+ sample + '/' + var + '/X_val_'+str(n)+'.npy',y='data/' + sample + '/' + var +'/y_val_'+str(n)+'.npy')
