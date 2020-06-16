import pandas as pd
import os
from datetime import datetime
import h5py 

def save_log(losses, accuracy,epoch,n,upperlower): 
  data = {'loss_train':losses['train'], 'loss_val':losses['valid'],'acc_train':accuracy['train'],'acc_val':accuracy['valid']}
  accloss = pd.DataFrame(data)
  now = datetime.now() 
  acclosslogdir = "save_model/{}/accloss/bin{}/".format(upperlower, n) + now.strftime("%Y%m%d-%H%M%S") + "/"
  os.mkdir(acclosslogdir)
  
  accloss.to_csv(acclosslogdir+"accloss.csv", mode='a', index=False)

def save_efficiency(tpr, fpr, n,upperlower):
    hf = h5py.File( "save_model/{}/efficiency/efficiency_B{}.h5".format(upperlower,n),'w')
    hf.create_dataset('tpr', data=tpr)
    hf.create_dataset('fpr', data=fpr)
    hf.close()
