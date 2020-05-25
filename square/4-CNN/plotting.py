import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc

def plottingLossAcc(losses,accuracy,Eb):
  # plotting losses, and accuracy
    plt.figure(figsize=(6.5,4.5))
    plt.plot(range(1,len(losses['train'])+1),losses['train'], 'r',label='train')
    plt.plot(range(1,len(losses['valid'])+1),losses['valid'], 'b',label='test')
    plt.title('Losses B={}'.format( Eb),fontsize=20)
    plt.xlabel('epoch',fontsize=15)
    plt.ylabel('losses',fontsize=15)
    plt.legend(fontsize=15,loc="upper right")
    plt.tick_params(axis = 'both', labelsize =15)
    plt.savefig("../plots/accloss/loss_B{}.pdf".format(Eb))
    #plt.show()

    plt.figure(figsize=(6.5,4.5))
    plt.plot(range(1,len(accuracy['train'])+1), accuracy['train'], 'r', label='train')
    plt.plot(range(1,len(accuracy['valid'])+1), accuracy['valid'], 'b', label='test')
    plt.title('Accuracy B={}'.format(Eb),fontsize=20)
    plt.xlabel('epoch',fontsize=15)
    plt.ylabel('accuracy',fontsize=15)
    plt.legend(fontsize=15,loc="lower right")
    plt.tick_params(axis = 'both', labelsize =15)
    plt.savefig("../plots/accloss/accuracy_B{}.pdf".format(Eb))
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
    fig.savefig("../plots/accloss/roc_B{}.pdf".format(Eb))
    #plt.show()
    print('Accuracy of the network on the {} test images: {} %'.format(len(testloader)*batchsize,100 * correct / total))
