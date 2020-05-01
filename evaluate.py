from ml import RatioEstimator
from ml.utils.loading import Loader

loading = Loader()
carl = RatioEstimator()
#do = "MUR1VsMUR2"
do = "sherpaVsMG5"
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
   
