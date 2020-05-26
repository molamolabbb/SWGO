import pandas as pd
import os
from datetime import datetime

def save_log(losses, accuracy,epoch,n): 
  accloss = pd.DataFrame([losses['train'],losses['valid'],accuracy['train'],accuracy['valid']],
             columns=list(range(epoch)),index=['loss_train','loss_val','acc_train','acc_val'])
  now = datetime.now() 
  acclosslogdir = "save_model/accloss/bin{}/".format(n) + now.strftime("%Y%m%d-%H%M%S") + "/"
  os.mkdir(acclosslogdir)
  
  accloss.to_csv(acclosslogdir+"accloss.csv", mode='a')
