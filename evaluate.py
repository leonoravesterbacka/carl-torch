from ml import RatioEstimator
from ml.utils.loading import Loader
loading = Loader()

carl = RatioEstimator()
do = "MUR1VsMUR2"
#do = "sherpaVsMG5"
carl.load('models/'+do+'_carl')
r_hat = carl.evaluate(x='data/'+do+'/x0_train.npy')
print("weights evaluated on training sample", r_hat)

loading.load_result(x0='data/'+do+'/x0_train.npy',     
                    x1='data/'+do+'/x1_train.npy',
                    weights=r_hat, 
                    label = 'train',
                    do = do,
)                                        

#r_hat = carl.evaluate(x='data/'+do+'/x0_test.npy')
#print("weights evaluated on test sample", r_hat)
#
#loading.load_result(x0='data/'+do+'/x0_test.npy',     
#                    x1='data/'+do+'/x1_train.npy',
#                    weights=r_hat, 
#                    label = 'test',
#                    do = do,
#)                 
#   
