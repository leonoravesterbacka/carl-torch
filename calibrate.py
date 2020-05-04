import numpy as np
from ml import RatioEstimator
from ml.utils.loading import Loader
from ml.calibration import CalibratedClassifier
from ml.base import Estimator
loading = Loader()

carl = RatioEstimator()
#do = "MUR1VsMUR2"
do = "sherpaVsMG5"
carl.load('models/'+do+'_carl')
#load
X = 'data/'+do+'/x_train.npy'
y = 'data/'+do+'/y_train.npy'
_, r_hat, _ = carl.evaluate(X)
print("uncalibrated weight is ",1/r_hat)
calib = CalibratedClassifier(carl)
calib.fit(X = X,y = y)
r = calib.predict(X ='data/'+do+'/x0_train.npy')
r[r == np.inf] = 1
w = 1/r
print("calibrated weight r is ", w)
loading.load_result(x0='data/'+do+'/x0_train.npy',     
                    x1='data/'+do+'/x1_train.npy',
                    weights=w, 
                    label = 'calibrated',
                    do = do,
)              
