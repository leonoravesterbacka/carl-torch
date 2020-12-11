import os
import sys
import logging
import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples',   action='store', type=str, dest='samples',   default='dilepton', help='samples to derive weights for. Sherpa 2.2.8 ttbar dilepton')
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='QSFUP', help='variation to derive weights for. default QSF down to QSF up')
parser.add_option('-n', '--nentries',  action='store', type=str, dest='nentries',  default=1000, help='specify the number of events do do the training on, default None means full sample')
parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='/eos/atlas/unpledged/group-tokyo/users/tatsuya/TruthAOD/Temp/Tuples/', help='path to where the data is stored')

(opts, args) = parser.parse_args()
sample = opts.samples
var = opts.variation
n = opts.nentries
p = opts.datapath
logger = logging.getLogger(__name__)
if os.path.exists('data/'+sample+'/'+var+'/X_train_'+str(n)+'.npy'):
    logger.info(" Doing evaluation of model trained with datasets: %s , generator variation: %s  with %s  events.", sample, var, n)
else:
    logger.info(" No datasets available for evaluation of model trained with datasets: %s , generator variation: %s  with %s  events.", sample, var, n)
    logger.info("ABORTING")
    sys.exit()
    
loading = Loader()
carl = RatioEstimator()
carl.load('models/'+sample+'/'+var+'_carl_'+str(n))
evaluate = ['train','val']
for i in evaluate:
    r_hat, _ = carl.evaluate(x='data/'+sample+'/'+var+'/X0_'+i+'_'+str(n)+'.npy')
    w = 1./r_hat
    loading.load_result(x0='data/'+sample+'/'+var+'/X0_'+i+'_'+str(n)+'.npy',     
                        x1='data/'+sample+'/'+var+'/X1_'+i+'_'+str(n)+'.npy',
                        weights=w, 
                        label=i,
                        do=sample,
                        var=var,
                        plot=True,
                        n=n,
                        path=p,
    )
carl.evaluate_performance(x='data/'+sample+'/'+var+'/X_val_'+str(n)+'.npy',y='data/'+sample+'/'+var+'/y_val_'+str(n)+'.npy')
