import numpy as np
def to_onehot(label):
    label=label.astype(int)
    num_class = len(set(label))
    label_set= np.eye((num_class))
    return label_set[label]

def lr_schedule(schedule,epoch):
    begin_epoch,end_epoch,lr_up,lr_low=schedule[0],schedule[1],schedule[2],schedule[3]
    if epoch<=begin_epoch:
        return lr_up
    elif epoch<=end_epoch:
        return lr_up-(epoch-begin_epoch)/(end_epoch-begin_epoch)*(lr_up-lr_low)
    else:
        return lr_low
    
class Sigmoid():
    def __init__(self):
        pass
    def forward(self,x):
        return 1./(1+np.exp(-x))
    def backward(self,x):
        # f'(x) = y(1-y)
        s=1./(1+np.exp(-x))
        return s*(1-s)

class Relu():
    def __init__(self):
        pass
    def forward(self,x):
        return (np.abs(x)+x)/2
    def backward(self,x):
        # f'(x) = 1 if x >=0 else 0
        return np.ones(x.shape)*(x>=0)

class Softmax():
    def __init__(self):
        pass
    def forward(self,x):
        a,b=x.shape
        m=x.max(axis=1).reshape((a,1))
        tmp=np.exp(x-m)
        s=np.sum(tmp,axis=1,keepdims=True)
        val=tmp/s
        return val
    def backward(self,x):
        m = np.diag(x)
        for i in range(len(m)):
            for j in range(len(m)):
                if i == j:
                    m[i][j]=x[i]*(1-x[i])
                else: 
                    m[i][j]=-x[i]*x[j]
        return m

class CrossEntropy():
    def __init__(self):
        pass
    def forward(self,yhat,y):
        yhat=np.clip(yhat,0.0001,0.9999)    
        loss=-np.mean(np.multiply(np.log(yhat),y)+np.multiply(np.log(1-yhat),(1-y)))
        return loss
    def backward(self,yhat,y):
        return (yhat-y)/(yhat*(1-yhat))

class CEwithLogit():
    def forward(self,logits,y):
        a,b=y.shape
        beta=logits.max(axis=1).reshape((a,1))
        tmp=logits-beta
        tmp=np.exp(tmp)
        tmp=np.sum(tmp,axis=1)
        tmp=np.log(tmp+1.0e-10)
        los=-np.sum(y*logits)+np.sum(beta)+np.sum(tmp)
        los=los/a
        return los
    def backward(self,logits,y):
        a,b=y.shape
        beta=logits.max(axis=1).reshape((a,1))
        tmp=logits-beta
        tmp=np.exp(tmp)
        numer=np.sum(tmp,axis=1,keepdims=True)
        yhat=tmp/numer
        der=(yhat-y)/a
        return der

def accuracy(y_hat,y):
    #计算预测出的label y_hat和正确的label y之间的预测accuracy
    n=y.shape[0]
    acc=np.sum(np.argmax(y_hat,axis=1)==np.argmax(y,axis=1))/n
    return acc