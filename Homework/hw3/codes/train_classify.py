import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet import ResNetLight
from tensorboardX import SummaryWriter
from torchsummary import summary

num_epoch = 400
learning_rate = 0.01
train_batch_size = 128
test_batch_size = 256
data_root = './data'
criterion = nn.CrossEntropyLoss()
num_workers = 10

writer = SummaryWriter('tensorboard/classify_log')
net = ResNetLight()
summary(net.cuda(), (3, 32, 32))
writer.add_graph(net, (torch.zeros(1, 3,32, 32).cuda()))
net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
# net = nn.DataParallel(ResNetLight(), device_ids=range(torch.cuda.device_count())).cuda()
fc = net.module.fc.parameters()
params = [p for n, p in net.named_parameters() if 'fc' not in n]
optimizer = optim.SGD([{'params': params}, {'params': fc, 'weight_decay': 5e-3}], lr=learning_rate, momentum=0.9, weight_decay = 1e-4)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=num_workers)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

tester = iter(testloader)
counter = 0

for epoch in range(num_epoch):
    if epoch == 50:
        learning_rate = 0.001
        optimizer = optim.SGD([{'params': params}, {'params': fc, 'weight_decay': 8e-3}], lr=learning_rate, momentum=0.9, weight_decay = 8e-4)

    if epoch == 120:
        learning_rate = 1e-4
        optimizer = optim.SGD([{'params': params}, {'params': fc, 'weight_decay': 1e-2}], lr=learning_rate, momentum=0.9, weight_decay = 1e-3)

    if epoch == 300:
        optimizer = optim.SGD([{'params': params}, {'params': fc, 'weight_decay': 1e-2}], lr=learning_rate, momentum=0.9, weight_decay = 5e-3)

    net.train()
    for i, batch in enumerate(trainloader):
        X, y = batch[0].to(torch.cuda.current_device()), batch[1].to(torch.cuda.current_device())
        optimizer.zero_grad()
        y_hat = net(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        counter += 1
        if i % 20 == 19:
            _, idx = torch.max(y_hat, 1)
            acc = torch.mean((idx == y).float()).item() * 100
            writer.add_scalar('train_loss', loss.item(), counter)
            writer.add_scalar('train_acc', acc, counter)
            print('[%d, %5d] loss: %.3f accuracy:%d' % (epoch + 1, i + 1, loss.item(), acc))

    net.eval()
    tester = iter(testloader)
    testbatch = next(tester)
    X, y = testbatch[0].to(torch.cuda.current_device()), testbatch[1].to(torch.cuda.current_device())
    y_hat = net(X)
    loss = criterion(y_hat, y)
    _, idx = torch.max(y_hat, 1)
    acc = torch.mean((idx == y).float()).item()*100
    print('test acc:',  acc, '%')
    writer.add_scalar('test_loss', loss.item(), counter)
    writer.add_scalar('test_acc', acc, counter)

    if epoch % 40 == 39:
        print('saving models')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 'checkpoint/resnetlight_classify_epoch' + str(epoch + 1) + '.model')
        print('model saved')
