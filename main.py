import utils,mlp_model,mnist_data
import random
import numpy as np
from matplotlib import pyplot as plt
import pickle
import time
import argparse
def train(model, X, Y, val_X, val_Y, batch_size, epoches, lr_schedule, weight_decay):
    #输入sample和各个参数，训练model参数
    max_acc = 0.
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in range(epoches):
        loss, acc = model.one_epoch(X, Y, batch_size, epoch,lr_schedule,weight_decay,train = True)
        val_loss, val_acc = model.one_epoch(val_X, val_Y, batch_size, 1,lr_schedule, weight_decay,train = False)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_acc > max_acc:
            pickle.dump(model,open("./Model_save/best_trained_model.dat","wb"))
        if epoch%10 == 0:
            print("Training step: Epoch {}/{}: Loss={}, Accuracy={}".format(epoch, epoches, loss, acc))
        train_losses.append(loss)
        train_accs.append(acc)
        
    return train_losses, train_accs, val_losses, val_accs, model.W


def parameter_sweep_hidden_layer():
    f = open("./hidden_layer_log.txt","w")
    #测试最佳hidden layer dim
    lr_sche=[100,200,0.1,0.01]  #lr初始设置为200个epoch都是0.1
    batch_size = 256
    epoches = 300
    weight_decay=0.001  #weight decay初始设置为0
    np.random.seed(4)
    # hidden_layer_list= np.linspace(10, 780, 10)
    hidden_layer_list_1 = np.linspace(20, 120, 6)
    hidden_layer_list_2 = np.linspace(10, 50, 5)
    v_acc_list=[]
    for i in hidden_layer_list_1:
        for j in hidden_layer_list_2:
            start_time = time.time()
            print("Hidden layer dim={} + {}".format(int(i),int(j)))
            model=mlp_model.MLP_model(int(i), int(j))
            trainX, trainY, valX, valY, testX, testY = mnist_data.load_dataset()
            trainX, valX, testX = trainX/255, valX/255, testX/255
            # problem may come from the function to_onehot
            trainY = utils.to_onehot(trainY)
            valY = utils.to_onehot(valY)
            testY = utils.to_onehot(testY)
            train_loss, train_acc, v_loss, v_acc, W= train(model, trainX, trainY, valX, valY, batch_size, epoches,lr_sche,weight_decay)
            v_acc_list.append(max(v_acc))
            f.write("Hidden layer dim={} + {}, max accuracy is {}".format(int(i),int(j),max(v_acc)) + "\n")
            end_time = time.time()
            used_time = end_time - start_time
            print(f"it comsumes {round(used_time,3)} s")
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x = [[i] * len(hidden_layer_list_2) for i in hidden_layer_list_1]
    y = [[i for i in hidden_layer_list_2] for _ in range(len(hidden_layer_list_1))]
    # x, y = np.meshgrid(x,y)
    Z = np.array(v_acc_list).reshape(len(hidden_layer_list_1), len(hidden_layer_list_2))
    ax.plot_surface(x, y, Z, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
    plt.savefig("./new_plots/sweep_hidden_layer.png")
    index_i = v_acc_list.index(max(v_acc_list)) // len(hidden_layer_list_2)
    index_j = v_acc_list.index(max(v_acc_list)) % len(hidden_layer_list_2)
    plt.show()    
    return hidden_layer_list_1[index_i], hidden_layer_list_2[index_j], max(v_acc_list)
def parameter_sweep_lr(hidden_layer_1,hidden_layer_2):
    #测试最佳learning rate，输入上一步骤选择的最佳hidden layer dim
    lr_max=[1e-4,1e-3,5e-3,0.01,0.05,0.1,0.5,1,5]
    v_acc_list=[]
    for i in lr_max:
        print("lr={}".format(i))
        lr_sche=[100,200,i,i/10]    #采用的learning rate schedule是前100个epoch保持lr_max，后100个epoch线性下降为lr_max/10
        batch_size = 256
        epoches = 300
        weight_decay=0  #weight decay参数还是设置为0
        np.random.seed(4)
        model=mlp_model.MLP_model(hidden_layer_1,hidden_layer_2)
        trainX, trainY, valX, valY, testX, testY = mnist_data.load_dataset()
        trainX, valX, testX = trainX/255, valX/255, testX/255
        trainY = utils.to_onehot(trainY)
        valY = utils.to_onehot(valY)
        testY = utils.to_onehot(testY)
        train_loss, train_acc,v_loss, v_acc, W= train(model, trainX, trainY,valX, valY, batch_size, epoches,lr_sche,weight_decay)
        # v_loss, v_acc = model.one_epoch(valX, valY, batch_size, 1,lr_sche,weight_decay,train = False)
        v_acc_list.append(max(v_acc))
    x = range(len(lr_max))
    fig = plt.figure(2)
    plt.xlabel("learning rate")
    plt.ylabel("val_acc after 300 epochs")
    plt.plot(x,v_acc_list,marker='o')
    plt.xticks(x,lr_max)
    plt.savefig('./new_plots/sweep_lr.png')
    plt.show()
    return lr_max[v_acc_list.index(max(v_acc_list))]

def parameter_sweep_weight_decay(hidden_layer_1,hidden_layer_2,lr):
    #测试最佳的正则化参数，输入上两个步骤选择的最佳hidden layer dim和learning rate
    lr_sche=[100,200,lr,lr/10]
    batch_size = 256
    epoches = 300
    weight_decay_list=[0.00001,0.00005,0.0001,0.0005,0.001,0.01]
    v_acc_list=[]
    np.random.seed(4)
    model=mlp_model.MLP_model(hidden_layer_1,hidden_layer_2)
    for i in weight_decay_list:
        print("weight decay parameter lambda={}".format(i))
        weight_decay=i
        trainX, trainY, valX, valY, testX, testY = mnist_data.load_dataset()
        trainX, valX, testX = trainX/255, valX/255, testX/255
        trainY = utils.to_onehot(trainY)
        valY = utils.to_onehot(valY)
        testY = utils.to_onehot(testY)
        train_loss, train_acc,v_loss, v_acc,W = train(model, trainX, trainY,valX, valY, batch_size, epoches,lr_sche,weight_decay)
        # v_loss, v_acc = model.one_epoch(valX, valY, batch_size, 1,lr_sche,weight_decay,train = False)
        v_acc_list.append(max(v_acc))
    x = range(len(weight_decay_list))
    fig = plt.figure(3)
    plt.xlabel("weight decay parameter lambda")
    plt.ylabel("val_acc after 300 epochs")
    plt.plot(x,v_acc_list,marker='o')
    plt.xticks(x,weight_decay_list)
    plt.savefig('./new_plots/sweep_weight_decay.png')
    plt.show()
    return weight_decay_list[v_acc_list.index(max(v_acc_list))]

def plot_model(hidden_layer_1, hidden_layer_2,lr,weight_decay):
    #对于调好参数的模型进行可视化和在test sample上的acc测试
    lr_sche=[100,200,lr,lr/10]
    batch_size = 256
    epoches = 300
    np.random.seed(4)
    model=mlp_model.MLP_model(hidden_layer_1, hidden_layer_2)
    trainX, trainY, valX, valY, testX, testY = mnist_data.load_dataset()
    trainX, valX, testX = trainX/255, valX/255, testX/255
    trainY = utils.to_onehot(trainY)
    valY = utils.to_onehot(valY)
    testY = utils.to_onehot(testY)
    train_loss, train_acc, val_loss, val_acc, W = train(model, trainX, trainY,valX, valY, batch_size, epoches,lr_sche,weight_decay)
    # print("Val: Loss={}, Accuracy={}".format(v_loss,v_acc))
    epoch_list=np.linspace(1, epoches, epoches)
    #training acc关于epoch的变化
    fig = plt.figure(4)
    plt.xlabel("epoch")
    plt.ylabel("training accuracy")
    plt.plot(epoch_list,train_acc)
    plt.savefig('./plot/train_acc.png')
    plt.show()
    #training loss关于epoch的变化
    fig = plt.figure(5)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.plot(epoch_list,train_loss)
    plt.savefig('./plot/train_loss.png')
    plt.show()
    fig = plt.figure(6)
    plt.xlabel("epoch")
    plt.ylabel("validating accuracy")
    plt.plot(epoch_list,val_acc)
    plt.savefig('./plot/val_acc.png')
    plt.show()
    fig = plt.figure(7)
    plt.xlabel("epoch")
    plt.ylabel("validating loss")
    plt.plot(epoch_list,val_loss)
    plt.savefig('./plot/val_loss.png')
    plt.show()
    
    #模型参数可视化
    fig = plt.figure(8)
    plt.axis('off')
    plt.imshow(W[0], cmap='RdBu')
    plt.savefig('./plot/weight_visualization_W1.png')
    plt.show()
    fig = plt.figure(9)
    plt.axis('off')
    plt.imshow(W[1], cmap='RdBu')
    plt.savefig('./plot/weight_visualization_W2.png')
    plt.show()
    #模型在test sample上的测试
    test_loss, test_acc = model.one_epoch(testX, testY, batch_size, 1,lr_sche,weight_decay,train = False)
    print("Test: Loss={}, Accuracy={}".format(test_loss,test_acc))
    return model

def load_model(dir):
    model = pickle.load(open(dir,"wb"))
    return model    
def Test(model):
    _, _, _, _, test_x, test_y = mnist_data.load_dataset()
    yhat = model.predict(test_x)
    accuracy = utils.accuracy(yhat, test_y)
    return accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on MNIST')
    parser.add_argument("--hidden_layer_1",type = int, default= 100,
                        help="dimention of the first hidden layer")
    parser.add_argument("--hidden_layer_2",type = int, default= 10,
                        help="dimention of the second hidden layer")
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument("--weight_decay", type=int, default=0.001,
                        help="weight decay parameter lambda")
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    # sweeping hidden layers
    hidden_layer_1, hidden_layer_2,best_accuracy = parameter_sweep_hidden_layer()
    print("hidden layer {} + {} yield the best accuracy".format(hidden_layer_1,hidden_layer_2,best_accuracy))
    
    # sweeping learning rate
    best_lr = parameter_sweep_lr(args.hidden_layer_1, args.hidden_layer_2)
    print("learning rate {} yield the best accuracy".format(best_lr))
    
    # sweeping weight decay
    best_weight_decay = parameter_sweep_weight_decay(args.hidden_layer_1, args.hidden_layer_2,args.lr)
    print(f"weight decay {best_weight_decay} yield the best accuracy")
    
    # visualization of the training process 
    m=plot_model(args.hidden_layer_1, args.hidden_layer_2 ,args.lr,args.weight_decay)
    pickle.dump(m,open("./Model_save/trained_model.dat","wb"))
    
    # test
    accuracy = Test(m)
    print("test accuracy: {}".fromat(accuracy))
    
    
if __name__ == "__main__":
    main()
