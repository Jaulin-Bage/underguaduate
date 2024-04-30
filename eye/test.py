# coding: utf-8
# mail: hph@mail.nwpu.edu.cn
# author: Jaulin_bage in 2022/12/30---17:22
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch.nn.functional as f
import timm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path= r'K:/database/eye_images'
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 3)

        self.fc1 = nn.Linear(4 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        input_size = x.size(0)
        # in: batch*1*24*24, out: batch*2*20*20(24-5+1)
        x = self.conv1(x)
        # out: batch*2*20*20
        x = f.relu(x)
        # in: batch*2*20*20, out: batch*2*10*10
        x = f.max_pool2d(x, 2, 2)

        # in: batch*2*10*10, out: batch*4*8*8 (10-3+1)
        x = self.conv2(x)

        x = f.relu(x)
        # batch*4*8*8

        # 4*8*8 = 256
        x = x.view(input_size, -1)

        # in: batch*256  out:batch*128
        x = self.fc1(x)
        x = f.relu(x)

        # in:batch*128 out:batch*2
        x = self.fc2(x)
        return f.log_softmax(x)

net=Net()
if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                             torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
    transforms = transforms.Compose([transforms.ToTensor()])  # 把图片进行归一化，并把数据转换成Tensor类型
    BATCH_SIZE = 256
    EPOCHS = 10
    # trainData = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)
    # testData = torchvision.datasets.MNIST('./data/', train=False, transform=transform)

    # trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    # testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)
    data_train = datasets.ImageFolder(path, transform=transforms)
    trainDataLoader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    data_test = datasets.ImageFolder(path, transform=transforms)
    testDataLoader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)


    print(net.to(device))

    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    history = {'Test Loss': [], 'Test Accuracy': []}
    for epoch in range(1, EPOCHS + 1):
        processBar = tqdm(trainDataLoader, unit='step')
        net.train(True)
        for step, (trainImgs, labels) in enumerate(processBar):
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)

            net.zero_grad()
            outputs = net(trainImgs)
            loss = lossF(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels) / labels.shape[0]
            loss.backward()

            optimizer.step()
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                       (epoch, EPOCHS, loss.item(), accuracy.item()))

            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)
                for testImgs, labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)

                    totalLoss += loss
                    correct += torch.sum(predictions == labels)
                testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
                testLoss = totalLoss / len(testDataLoader)
                history['Test Loss'].append(testLoss.item())
                history['Test Accuracy'].append(testAccuracy.item())
                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                           (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(),
                                            testAccuracy.item()))
        processBar.close()

    plt.plot(history['Test Loss'], label='Test Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.show()

    plt.plot(history['Test Accuracy'], color='red', label='Test Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('acc.png')
    plt.show()

    torch.save(net, './model.pth')

