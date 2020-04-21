from ml import RatioEstimator
from ml.utils.loading import Loader
loading = Loader()

carl = RatioEstimator()
carl.load('models/carl')

r_hat = carl.evaluate(x='data/samples/x0_train.npy')
print("weights evaluated on training sample", r_hat)

loading.load_result(x0='data/samples/x0_train.npy',     
                    x1='data/samples/x1_train.npy',
                    weights=r_hat, 
                    label = 'train')                                        

r_hat = carl.evaluate(x='data/samples/x0_test.npy')
print("weights evaluated on test sample", r_hat)

loading.load_result(x0='data/samples/x0_test.npy',     
                      x1='data/samples/x1_train.npy',
                      weights=r_hat, 
                      label = 'test')                 
   
