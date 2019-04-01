import torch
import torch.optim as optim
import torch.nn as nn
from rotdataset import Rot_Dataset
from resnet import ResNetLight
from PIL import Image
from torchvision import transforms

train_batch_size = 512
num_workers = 10
criterion = nn.CrossEntropyLoss(reduce=False)
classes = ['up', 'left', 'down', 'right']

check = torch.load('checkpoint/resnetlight_rotate_epoch80.model')
net = ResNetLight(len(classes)).cuda()
net.load_state_dict(check['model_state_dict'])
net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
net.eval()
testloader = torch.utils.data.DataLoader(Rot_Dataset(train = False), batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
toPIL = transforms.ToPILImage()

maxloss = -1
maxX = None
minloss = 100
minX = None
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std= torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1 , 1)

for i, batch in enumerate(testloader):
    X, y = batch[0].to(torch.cuda.current_device()), batch[1].to(torch.cuda.current_device())
    y_hat = net(X)
    loss = criterion(y_hat, y)
    value, idx = torch.max(loss, 0)
    if value > maxloss:
        maxloss = value
        img = X[idx].detach().cpu() * std + mean
        maxX = toPIL(img)

    value, idx = torch.min(loss, 0)
    if value < minloss:
        minloss = value
        img = X[idx].detach().cpu() * std + mean
        minX = toPIL(img)

maxX.save('max.jpg')
minX.save('min.jpg')

