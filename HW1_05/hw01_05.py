from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from torch.autograd import Variable
import matplotlib as mpl
import numpy as np
from torchvision import models	
import sys
import random
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES']='0'
PATH = "./Model/model.pth"
EPOCH = 30
BATCH_SIZE = 32
TRAIN_NUMS = 49000
PRINT_FREQ = 100
learning_rate = 0.001

'''
print(torch.version.cuda)  
print(torch.cuda.device_count())
print(torch.cuda.is_available())
'''

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
label_dict={0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()

data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader_all = torch.utils.data.DataLoader(test_data, batch_size=5500, shuffle=True)

val_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(TRAIN_NUMS, 50000)))

def flatten(x):    
    x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
    return x

class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer

        self.loss_list = []
        self.iter_loss_list = []
        
        self.train_acc_list = []
        self.test_acc_list = []
        
        self.device = device
        self.acc = 0
        self.loss = 0

    def train_loop(self, model, train_loader, val_loader):
        self.train_acc_list.clear()
        self.loss_list.clear()

        for epoch in range(EPOCH):
            print("\n---------------- Epoch [{}] ----------------".format(epoch+1))
            self.train_part(model, train_loader, epoch)
            self.validate(model, val_loader, epoch)
            self.validate(model, test_loader, epoch, state="Testing")
            
    def train_part(self, model, loader, epoch):
        model.train()
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            self.iter_loss_list.append(loss)

            if step >= 0 and (step % PRINT_FREQ == 0):
                self.logging(outs, y, loss, step, epoch, "Training")
            
            loss.backward()
            self.optimizer.step()
            
    def validate(self, model, loader, epoch, state="Validate"):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                
                outs = model(X)
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            y = torch.cat(y_list)
            self.logging(outs, y, loss, step, epoch, state)

        if state == "Testing":
            self.test_acc_list.append(self.acc)
        else:
            self.train_acc_list.append(self.acc)
            self.loss_list.append(self.loss)
        self.acc = 0
        self.loss = 0                
                
    def logging(self, output, target, loss, step, epoch, state):
        prediction = output.argmax(1)
        batch_size = target.size(0)
        correct = prediction.eq(target)
        accuracy = correct.float().sum(0) / batch_size
        print("Epoch [{:3d}/{}] {} Progress {:04d}, Loss = {:.3f}, Acc = {:.3f}".format(epoch+1, EPOCH, state, step, loss, accuracy))
        self.acc = accuracy
        self.loss = loss


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)        # 64*32*32
        self.conv2 = nn.Conv2d(64,64,3,padding=1)       # 64*32*32
        self.pool1 = nn.MaxPool2d(2, 2)                 # 64*16*16
        self.bn1 = nn.BatchNorm2d(64)                   # 64*16*16
        self.relu1 = nn.ReLU()   

        self.conv3 = nn.Conv2d(64,128,3,padding=1)      # 128*16*16
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)   # 128*16*16
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)      # 128*9*9
        self.bn2 = nn.BatchNorm2d(128)                  # 128*9*9
        self.relu2 = nn.ReLU()                          # 128*9*9

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)    # 128*9*9
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)   # 128*9*9
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)   # 128*11*11
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)      # 128*6*6
        self.bn3 = nn.BatchNorm2d(128)                  # 128*6*6
        self.relu3 = nn.ReLU()                          # 128*6*6

        self.fc14 = nn.Linear(128*6*6,128)         
        self.drop1 = nn.Dropout2d()                  
        self.fc15 = nn.Linear(128,64)              
        self.drop2 = nn.Dropout2d()                  
        self.fc16 = nn.Linear(64,10)                  
    
    def forward(self, x):
        x = x.to(device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x.view(-1,128*6*6)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate, momentum=0.9)

def show_image_and_label(images,labels,start):
    print('=========================== Q5-1 ===============================')
    fig = plt.gcf()
    fig.canvas.set_window_title('Q5-1')
    fig.set_size_inches(10, 5)
    fig.patch.set_facecolor('xkcd:salmon')
    for i in range(0, 10):
        sub = plt.subplot(2, 5, i+1)
        sub.set_xticks([]) 
        sub.set_yticks([])
        sub.imshow(images[start])
        sub.set_title(label_dict[labels[start][0]], fontsize=12)
        start += 1
    plt.savefig('image_n_label.png')
    plt.show()

def show_hyperparameters():
    print('=========================== Q5-2 ===============================')
    print('hyperparameters:')
    print('batch size: ', BATCH_SIZE)
    print('learning rate: ', learning_rate)
    print('optimizer:', optimizer)

def show_summary():
    print('=========================== Q5-3 ===============================')
    summary(model, (3,32,32))

def train():
    global EPOCH,model
    EPOCH = 30
        
    trainer = Trainer(criterion, optimizer, device)
    trainer.train_loop(model, train_loader, val_loader)

    tr_acc = trainer.train_acc_list
    te_acc = trainer.test_acc_list
    loss_list = trainer.loss_list
    
    train_accuracy_result = []
    test_accuracy_result = []
    loss = []

    for item in tr_acc:
        train_accuracy_result.append(item.item())
    for item in te_acc:
        test_accuracy_result.append(item.item())
    for item in loss_list:
        loss.append(item.item())
        
    train_accuracy_result = [i * 100 for i in train_accuracy_result]
    test_accuracy_result = [i * 100 for i in test_accuracy_result]
    torch.save(model.state_dict(), PATH) # save model

    plt.ylim(0, 100)
    plt.plot(range(1, EPOCH+1), train_accuracy_result, label='train')
    plt.plot(range(1, EPOCH+1), test_accuracy_result, label='test')
    plt.xlabel('epoch')
    plt.ylabel('%')
    plt.title('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('accuracy.png')
    plt.clf() 
    plt.plot(range(1, EPOCH+1), loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('loss.png')


def show_inference():
    print('=========================== Q5-5 ===============================')
    loaded_model = Net()
    loaded_model.load_state_dict(torch.load(PATH))
    loaded_model.eval()
    loaded_model.cuda()

    batch_index, (data, labels) = next(enumerate(test_loader_all))
    data = data.to(device)
    data = Variable(data, requires_grad=True)
    with torch.no_grad():
        output = loaded_model(data)
    
    num = input('input test image index (0-5499) :')
    num = int(num)
    probability = F.softmax(output.data[num], dim=0).tolist()
         
    plt.clf()       
    plt.bar(classes, probability)
    plt.savefig('bar-chart.png')
    plt.show()
    print('Finish')

# 5-1
rand1 = random.randint(0, 5000)
show_image_and_label(x_train_image, y_train_label, rand1)
# 5-2
show_hyperparameters()
# 5-3
show_summary()
# 5-4
print('=========================== Q5-4 ===============================')
run = input('Do you wanna train model ? (y/n) ')
if run == 'y':
    train()
else:
    pass
# 5-5
show_inference()