import logging
import optparse
import skorch
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
import scipy.stats as stats

from torch import nn
import torch
from torch import optim
import numpy as np
from ml import RatioEstimator
from ml import Loader
from ml.utils.tools import load
from ml.utils.tools import load_and_check

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='samples to derive weights for. default Sherpa vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples



class RatioModel(nn.Module):

    def __init__(self, n_observables=6, n_hidden=(4, 2), activation="relu", dropout_prob=0.5):

        super(RatioModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = "relu"
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables
        # Hidden layers
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                self.layers.append(nn.Dropout(self.dropout_prob))
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
        self.layers.append(nn.Linear(n_last, 1))

    def forward(self, x):
        s_hat = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                s_hat = torch.relu(s_hat)
            s_hat = layer(s_hat)
        s_hat = torch.sigmoid(s_hat)
        #r_hat = (1 - s_hat) / s_hat

        return s_hat


doWeights = True #if False, train model for hyperparameter search
loading = Loader()
x,y =loading.loading(
    folder='./data/',
    plot=True,
    do = do,
    randomize = False,
    save = True,
    correlation = False,
)
net = NeuralNetBinaryClassifier(
    RatioModel,
    criterion=nn.BCELoss,
    iterator_train__shuffle=True,
    optimizer = optim.Adam,
)

x = x.astype(np.float32)
print("x",x)
#scaler_x = MinMaxScaler()
scaler_x = RobustScaler()
scaler_x.fit(x)
x = scaler_x.transform(x)

print("y",len(y))
print("x dtype",x.dtype)
y = y.squeeze(1)
print("y",y)
print("x",len(y))
print("scaler x",x)
net.set_params(train_split=False, verbose=0)
print(net.get_params().keys())
params = {
    'lr': [0.001, 0.01],
    #'lr': [0.0001, 0.001,0.01,0.1, 1],
    #'optimizer': [optim.Adam, optim.SGD],
    #'batch_size':[32,64,128],
}
#gs = RandomizedSearchCV(net, params, refit = False,random_state=1, n_iter = 20, cv=3, scoring='accuracy', verbose=2)
gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy', verbose=2)
gs.fit(x, y)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
y_pred = gs.predict(x)
print("y_pred", y_pred)
print("Accuracy score %s" %accuracy_score(y,y_pred))


