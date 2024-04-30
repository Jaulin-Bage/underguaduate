# coding: utf-8
# mail: hph@mail.nwpu.edu.cn
# author: Jaulin_bage in 2023/2/28---10:43
import torch
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from test import Net
from tqdm import tqdm

pthfile = 'K:/documents/Big_duang/eye/model.pth'
path = r'K:/database/eye_images'
net = torch.load(pthfile, map_location=torch.device('cpu'))

if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                             torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
    transforms = transforms.Compose([transforms.ToTensor()])  # 把图片进行归一化，并把数据转换成Tensor类型
    BATCH_SIZE = 256
    EPOCHS = 32
    # trainData = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)
    # testData = torchvision.datasets.MNIST('./data/', train=False, transform=transform)

    # trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    # testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)

    data_test = datasets.ImageFolder(path, transform=transforms)
    testDataLoader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

    # print(net.to(device))

    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    history = {'Test Loss': [],'Test Accuracy': []}
    for epoch in range(1, EPOCHS + 1):
        processBar = tqdm(testDataLoader, unit='step')
        net.train(False)
        for step, (trainImgs, labels) in enumerate(processBar):
            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
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
                processBar.set_description("[%d/%d] Test Loss: %.4f, Test Acc: %.4f" %
                                           (epoch, EPOCHS, testLoss.item(),
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
