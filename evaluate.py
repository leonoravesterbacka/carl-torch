import os
import sys
import logging
import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader

#################################################
# Arugment parsing
parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-n', '--nominal',   action='store', type=str, dest='nominal',   default='', help='Nominal sample name (root file name excluding the .root extension)')
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='', help='Variation sample name (root file name excluding the .root extension)')
parser.add_option('-e', '--nentries',  action='store', type=str, dest='nentries',  default=1000, help='specify the number of events to do the training on, None means full sample')
parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='./Inputs/', help='path to where the data is stored')
parser.add_option('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
parser.add_option('-f', '--features',  action='store', type=str, dest='features',  default='', help='Comma separated list of features within tree')
parser.add_option('-w', '--weightFeature',  action='store', type=str, dest='weightFeature',  default='', help='Name of event weights feature in TTree')
parser.add_option('-t', '--TreeName',  action='store', type=str, dest='treename',  default='Tree', help='Name of TTree name inside root files')
(opts, args) = parser.parse_args()
nominal  = opts.nominal
variation = opts.variation
n = opts.nentries
p = opts.datapath
global_name = opts.global_name
features = opts.features.split(",")
weightFeature = opts.weightFeature
treename = opts.treename
#################################################


logger = logging.getLogger(__name__)
if os.path.exists('data/'+global_name+'/X_train_'+str(n)+'.npy') and os.path.exists('data/'+global_name+'/metaData_'+str(n)+'.pkl'):
    logger.info(" Doing evaluation of model trained with datasets: [{}, {}], with {} events.".format(nominal, variation, n))
else:
    logger.info(" No datasets available for evaluation of model trained with datasets: [{},{}] with {} events.".format(nominal, variation, n))
    logger.info("ABORTING")
    sys.exit()
    
loading = Loader()
carl = RatioEstimator()
carl.load('models/'+global_name+'_carl_'+str(n))
evaluate = ['train','val']
for i in evaluate:
    print("<evaluate.py::__init__>::   Running evaluation for {}".format(i))
    r_hat, _ = carl.evaluate(x='data/'+global_name+'/X0_'+i+'_'+str(n)+'.npy')
    w = 1./r_hat
    print("<evaluate.py::__init__>::   Loading Result for {}".format(i))
    loading.load_result(x0='data/'+global_name+'/X0_'+i+'_'+str(n)+'.npy',     
                        x1='data/'+global_name+'/X1_'+i+'_'+str(n)+'.npy',
                        metaData='data/'+global_name+'/metaData_'+str(n)+'.pkl',
                        weights=w, 
                        features=features,
                        weightFeature=weightFeature,
                        label=i,
                        plot=True,
                        nentries=n,
                        TreeName=treename,
                        pathA=p+nominal+".root",
                        pathB=p+variation+".root",
                        global_name=global_name,
                    )
# Evaluate performance
carl.evaluate_performance(x='data/'+global_name+'/X_val_'+str(n)+'.npy',
                          y='data/'+global_name+'/y_val_'+str(n)+'.npy')
