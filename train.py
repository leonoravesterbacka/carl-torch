import logging
from ml import RatioEstimator
from ml import Loader

loading = Loader()
do = 'MUR1VsMUR2'
#do = 'sherpaVsMG5'
x, y = loading.loading(
    folder='./data/',
    filename='train',
    plot=True,
    do = do,
)
estimator = RatioEstimator(
    n_hidden=(10,10),
    activation="sigmoid"
)
estimator.train(
    method='carl',
    batch_size = 128,
    x='data/'+do+'/x_train.npy',
    y='data/'+do+'/y_train.npy',
    scale_inputs = True,
)

estimator.save('models/'+do+'_carl')
