import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='chose what samples to perform weight derivation on. default Sherpa     vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples

loading = Loader()
carl = RatioEstimator()
carl.load('models/'+do+'_carl')
_, r_hat, _ = carl.evaluate(x='data/'+do+'/x0_train.npy')
w = 1./r_hat
print("weights evaluated on training sample, uncalibrated", w)

loading.load_result(x0='data/'+do+'/x0_train.npy',     
                    x1='data/'+do+'/x1_train.npy',
                    weights=w, 
                    label = 'train',
                    do = do,
)   
                 
_, r_hat, _ = carl.evaluate(x='data/'+do+'/x0_test.npy')
w = 1./r_hat
print("weights evaluated on test sample, uncalibrated", w)

loading.load_result(x0='data/'+do+'/x0_train.npy',     
                    x1='data/'+do+'/x1_train.npy',
                    weights=w, 
                    label = 'test',
                    do = do,
)                    
