import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc

def plottingLossAcc(losses, accuracy, Eb, limitsloss, limitsacc, bestacc, epoch):
  # plotting losses, and accuracy
    plt.figure(figsize=(6.5,4.5))
    plt.plot(range(1,len(losses['train'])+1),losses['train'], 'r',label='train')
    plt.plot(range(1,len(losses['valid'])+1),losses['valid'], 'b',label='test')
    plt.title('Losses B={}'.format( Eb),fontsize=20)
    plt.xlabel('epoch',fontsize=15)
    plt.ylabel('losses',fontsize=15)
    plt.legend(fontsize=15,loc="upper right")
    plt.tick_params(axis = 'both', labelsize =15)
    plt.axis(limitsloss)
    plt.grid(True)
    plt.savefig("../plots/accloss/loss_B{}.pdf".format(Eb),bbox_inches='tight')
    #plt.show()

    plt.figure(figsize=(6.5,4.5))
    plt.plot(range(1,len(accuracy['train'])+1), accuracy['train'], 'r', label='train')
    plt.plot(range(1,len(accuracy['valid'])+1), accuracy['valid'], 'b', label='test')
    plt.title('Accuracy B={}'.format(Eb),fontsize=20)
    plt.xlabel('epoch',fontsize=15)
    plt.ylabel('accuracy',fontsize=15)
    plt.legend(fontsize=15,loc="lower right")
    plt.tick_params(axis = 'both', labelsize =15)
    plt.axis(limitsacc)
    plt.text(epoch/2,0.58,'Accuracy : {}%'.format(bestacc), alpha=1,
                va="center", ha="center", fontsize=15)
    plt.grid(True)
    plt.savefig("../plots/accloss/accuracy_B{}.pdf".format(Eb),bbox_inches='tight')
    #plt.show()

def roc(y,y_score,correct,total,Eb):
    ny = y
    fpr,tpr, thresholds  = roc_curve(np.array(ny).ravel(),y_score.ravel())
    roc_auc = auc(fpr,tpr)
    fig, ax = plt.subplots(figsize=(6,6))
    lw = 2
    res = 1
    for i, (f, t) in enumerate(zip(fpr, tpr)):
        if abs(1-(f+t))<res:
            res = 1-(f+t)
            idx = i
    ax.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0,1.0])
    ax.set_title('ROC curve B={}'.format(Eb), fontsize=20)
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)
    ax.tick_params(axis = 'both', labelsize =15)
    ax.legend(loc="lower right",fontsize=15)
    t = ax.text(0.8, 0.15,'Accuracy : {}%'.format(100 * correct / total), alpha=1,
                 va="center", ha="center", size=15, transform=ax.transAxes)
    fig.savefig("../plots/accloss/roc_B{}.pdf".format(Eb),bbox_inches='tight')
    #plt.show()
    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
    return fpr,tpr, thresholds

def Significance(thresholds, outputs, labels):
    gamma = []
    proton = []
    for os,ls in zip(outputs, labels):
        for o, l in zip(os,ls):
            if l==1:
                gamma.append(o[0])
            else:
                proton.append(o[0])
    
    gamma = np.array(gamma)
    proton = np.array(proton)
    significance = []
    newthreshold = []
    for i,t in enumerate(sorted(thresholds)[:-1]):
        if (len(gamma[gamma>t])+len(proton[proton>t])) == 0:
            continue               
        significance.append(len(gamma[gamma>t])/((len(gamma[gamma>t])+len(proton[proton>t])*1000)**0.5))
        newthreshold.append(t)
    return newthreshold, significance

def Draw_Eff_Sig(fpr, tpr, significance, thresholds, newthresholds, n):
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    signi = ax.plot(newthresholds, significance, "g-", label= r'$\frac{s}{\sqrt{s+b}}$')
    
    tax = ax.twinx()
    txy = zip(tpr,thresholds)
    txy = sorted(txy, key=lambda b:b[1],reverse=True)
    tpr, thresholds = zip(*txy)
    Teff = tax.plot(sorted(thresholds)[:-1], tpr[:-1], "r-", label=r'$SigEff$') 

    fax = ax.twinx()
    fxy = zip(fpr,thresholds)
    fxy = sorted(fxy, key=lambda b:b[1],reverse=True)
    fpr, thresholds = zip(*fxy)
    Feff = fax.plot(sorted(thresholds)[:-1], fpr[:-1], "b-", label=r'$\frac{1}{BgEff}$')
    
    fax.spines['right'].set_position(('axes',1.2))
  
    ax.set_xlim(-0.05,1.05)
    tax.set_ylim(0,1.05)
    fax.set_ylim(min(fpr),max(fpr)*1.05)
    ax.set_ylim(min(significance)*0.9, max(significance)*1.05)
   
    ax.set_xlabel("cut value",fontsize=15)
    tax.set_ylabel(r"$SigEff$",fontsize=15)
    fax.set_ylabel(r'$\frac{1}{BG Eff}$',fontsize=20)
    ax.set_ylabel(r"$Significance$",fontsize=15) 

    tax.yaxis.label.set_color(Teff[0].get_color())
    fax.yaxis.label.set_color(Feff[0].get_color())
    ax.yaxis.label.set_color(signi[0].get_color())
     
    ax.tick_params(axis='x', labelsize = 15)
    tax.tick_params(axis='y' ,colors=Teff[0].get_color(), labelsize = 15)
    fax.tick_params(axis='y' ,colors=Feff[0].get_color(), labelsize = 15)
    ax.tick_params(axis='y', colors=signi[0].get_color(), labelsize = 15) 
    
    lines = [Teff[0],Feff[0],signi[0]]

    ax.legend(lines, [l.get_label() for l in lines], loc = 'upper left', fontsize=15)
    ax.set_title('Significance and Efficiency B={}'.format(n),fontsize=20)
    
    plt.grid(True)
    fig.savefig("../plots/output/significant_B{}.pdf".format(n),bbox_inches='tight')
    plt.show()

def CnnOutputs(outputs, labels, n):
    gamma = []
    proton = []

    for os,ls in zip(outputs, labels):
        for o, l in zip(os,ls):
            if l==1:
                gamma.append(o[0])
            else:
                proton.append(o[0])
                
    fig, ax = plt.subplots(figsize=(6,6))
    bins = np.arange(0,1.05,0.05)
    
    particle = 'proton'
    phist = ax.hist(proton,bins=bins,alpha=0.6,label=particle)
    
    particle = 'gamma'
    ghist = ax.hist(gamma,bins=bins,alpha=0.6,label=particle)
    
    ax.set_title('CNN Outputs',fontsize=20)
    ax.set_xlabel("sigmoid output",fontsize=15)
    ax.tick_params(axis = 'both', labelsize =15)
    ax.legend(fontsize=15)
    
    plt.grid(True)
    fig.savefig("../plots/output/cnn_output_B{}.pdf".format(n),bbox_inches='tight')
    plt.show()
