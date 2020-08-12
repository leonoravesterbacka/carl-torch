import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='samples to derive weights for. default Sherpa vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples

loading = Loader()
carl = RatioEstimator()
carl.load('models/'+do+'_carl')
evaluate = ['train']
for i in evaluate:
    r_hat, _ = carl.evaluate(x='data/'+do+'/x0_'+i+'.npy')
    w = 1./r_hat
    loading.load_result(x0='data/'+do+'/x0_'+i+'.npy',     
                        x1='data/'+do+'/x1_train.npy',
                        weights=w, 
                        label = i,
                        do = do,
                        save = True,
    )
loading = Loader()
carl.load('models/'+do+'_carlval')
carl.evaluate_performance(x='data/'+do+'/X_val.npy',y='data/'+do+'/Y_val.npy')
