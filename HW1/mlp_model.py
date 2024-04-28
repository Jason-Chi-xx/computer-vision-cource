import utils
import numpy as np
class MLP_model():
    def __init__(self,hidden_layer_1, hidden_layer_2):
        self.layer_dim=[784,hidden_layer_1, hidden_layer_2,10]
        self.num_layer=len(self.layer_dim)
        self.activation=[utils.Relu(),utils.Relu(),utils.Relu(),utils.Softmax()]
        self.loss=utils.CEwithLogit()
        self.W=[]
        self.b=[]
        for i in range(self.num_layer-1):
            W=np.random.randn(self.layer_dim[i],self.layer_dim[i+1])
            b=np.random.randn(1,self.layer_dim[i+1])
            self.W.append(W)
            self.b.append(b)
        self.A=[]   #A is the output through the activation, including original x
        self.Z=[]  #Z is the output of each layer (W*x+b)
    
    def forward_step(self,x):
        A,Z=[],[]
        for i in range(self.num_layer):
            if i==0:
                a=x.copy()
                z=x.copy()
            else:
                z=A[-1].dot(self.W[i-1])+self.b[i-1]
                a=self.activation[i].forward(z)
            A.append(a)
            Z.append(z)
        self.A=A
        self.Z=Z
        return Z[-1],A[-1]
    
    def backward_step(self, dl_dyhat):
        dl_dz_list=[]
        for i in range(self.num_layer-1,-1,-1):
            if i==self.num_layer-1:
                dl_dz=dl_dyhat
            else:
                dl_da=np.dot(dl_dz,self.W[i].T)
                dl_dz=self.activation[i].backward(self.Z[i])*dl_da
            dl_dz_list.append(dl_dz)
        dl_dz_list=list(reversed(dl_dz_list))   
        self.dl_dz_list=dl_dz_list
        return

    def update_weights(self,lr,weight_decay):
        for i in range(self.num_layer-1):
            a=self.A[i] 
            dW=np.dot(a.T,self.dl_dz_list[i+1])+2*weight_decay*self.W[i]
            db=np.sum(self.dl_dz_list[i+1],axis=0,keepdims=True)
            self.W[i]-=lr*dW
            self.b[i]-=lr*db
        return
    
    def one_epoch(self,X,Y,batch_size,epoch,lr_sche,weight_decay,train):
        n = X.shape[0]
        permutation = list(np.random.permutation(n))
        shuffled_x, shuffled_y = X[permutation, :], Y[permutation, :]
        loss_value, acc_value = 0, 0
        for i in range(0,len(shuffled_x),batch_size):
            x, y = shuffled_x[i:i+batch_size,:], shuffled_y[i:i+batch_size,:]
            z,yhat = self.forward_step(x)
            if train:
                dl_dz = self.loss.backward(z,y)
                self.backward_step(dl_dz)
                lr=utils.lr_schedule(lr_sche,epoch)
                self.update_weights(lr,weight_decay)
            loss_value += self.loss.forward(yhat,y)*x.shape[0]
            acc_value += utils.accuracy(yhat, y)*x.shape[0]
        loss_value = loss_value/n
        acc_value = acc_value/n
        return loss_value, acc_value
    
    def predict(self,x):
        z,yhat = self.forward_step(x)
        return yhat