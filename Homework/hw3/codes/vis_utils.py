# this pieces of code refered the script in https://github.com/metalbubble/CAM
from PIL import Image
from torchvision import models, transforms
from torch.nn import functional as F
import torch
import numpy as np
import cv2
from resnet import ResNetLight
import torchvision
from rotdataset import Rot_Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def back(t):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    return t * std + mean

class cam_renderer:
    def __init__(self, checkpoint, classes, ifrotate = False):
        check = torch.load(checkpoint)
        net = ResNetLight(len(classes)).cuda()
        net.load_state_dict(check['model_state_dict'])
        net.eval()

        normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
        )

        self.preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
        ])

        self.features = []

        def hook_feature(module, input, output):
            self.features.append(output.data.cpu().numpy())

        net.cpu()
        net._modules.get('resblocks4').register_forward_hook(hook_feature)

        params = list(net.parameters())
        self.net = net
        self.weight_fc = np.squeeze(params[-2].data.numpy())
        self.classes = classes
        self.ifrotate = ifrotate
        
    def returnCAM(self, feature, weight_softmax, class_idx):
        size_upsample = (256, 256)
        bz, nc, h, w = feature.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def render(self, img_name, outputpath = ''):
        print('doing ' + img_name)
        img_pil = Image.open(img_name)
        if self.ifrotate:
            angle = 90 * int(np.random.randint(4))
            img_pil = img_pil.rotate(angle, expand=True)
        img_tensor = self.preprocess(img_pil).unsqueeze(0)
        # img_variable = Variable(img_tensor)
        self.features = []
        logit = self.net(img_tensor)
        y_hat = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = y_hat.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        CAMs = self.returnCAM(self.features[0], self.weight_fc, [idx[0]])
        img = np.asarray(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        result = np.concatenate((img, result), axis=1)
        text = '{:.3f} -> {}'.format(probs[0], self.classes[idx[0]])
        cv2.putText(result, text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5)
        cv2.imwrite(outputpath + img_name.split('/')[-1][:-4] + '_CAM.jpg', result)

def render_filter(checkpoint):
    if 'classify' in checkpoint:
        nettype = 'classify'
    else:
        nettype = 'rotate'
    check = torch.load(checkpoint)

    statedict = check['model_state_dict']
    layer0 = statedict['start.0.weight']
    filters0 = layer0.permute(0, 2, 3, 1).cpu().numpy()
    filters0 = filters0 - np.min(filters0)
    filters0 = filters0 / np.max(filters0)
    s = 64
    img_cat = Image.new('RGB', (s * 8, s * 4))
    for i in range(32):
        f = filters0[i]
        img = Image.fromarray(np.uint8(f * 255)).resize((s -4, s-4))
        img_cat.paste(img, (i//4 * s + 2, i%4 * s + 2))
    img_cat.save('images/'+  nettype +'net_results/vis_filter/layer0.jpg')

    layer1 = statedict['resblocks1.0.conv.0.weight']
    filters1 = layer1.permute(0, 2, 3, 1).cpu().numpy()
    filters1 = filters1 - np.min(filters1)
    filters1 = filters1 / np.max(filters1)
    s = 64
    img_cat = Image.new('L', (s * 8, s * 4))
    for i in range(32):
        f = filters1[i][:,:,0]
        img = Image.fromarray(np.uint8(f * 255)).resize((s -4, s-4))
        img_cat.paste(img, (i//4 * s + 2, i%4 * s + 2))
    img_cat.save('images/'+  nettype +'net_results/vis_filter/layer1.jpg')


def render_knn(checkpoint, classes):
    check = torch.load(checkpoint)
    net = ResNetLight(len(classes)).cuda()
    net.load_state_dict(check['model_state_dict'])
    net.eval()
    features = []

    def hook_feature(module, input, output):
        features.append(output)

    net._modules.get('avgpool').register_forward_hook(hook_feature)

    train_batch_size = 100
    test_batch_size = 5
    data_root = './data'
    num_workers = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if 'classify' in checkpoint:
        nettype = 'classify'
        diffidx = [1, 3, 4, 6, 8]
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_test)
        trainloader = iter(torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers))
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    else:
        nettype = 'rotate'
        diffidx = [4, 14, 18, 27, 32]
        trainloader = iter(torch.utils.data.DataLoader(Rot_Dataset(train=True), batch_size=train_batch_size, shuffle=True, num_workers=num_workers))
        testset = Rot_Dataset(train=False)

    X_train, y_train = next(trainloader)
    X_train = X_train.cuda()
    X_test = torch.cat([testset[i][0].view(1, 3, 32, 32) for i in diffidx], 0)
    X_test = X_test.cuda()
    features = []
    net(X_train)
    net(X_test)
    feat_train = features[0].view(1, train_batch_size, 256)
    feat_test = features[1].view(test_batch_size, 1, 256)
    _, idx = torch.topk(torch.sum((feat_train - feat_test) ** 2, dim=2), 10, dim=1, largest=False)
    idx = idx.view(10 * test_batch_size)
    result = torch.index_select(X_train, 0, idx)
    toPIL = transforms.ToPILImage()

    result = back(result.detach().cpu())
    s = 100
    img_cat = Image.new('RGB', (s * 11 + 20, s * 5))

    for i in range(50):
        img = toPIL(result[i]).resize((s, s))
        img_cat.paste(img, ((i % 10 + 1) * s + 20, i // 10 * s))

    X_test = back(X_test.detach().cpu())
    for i in range(5):
        img = toPIL(X_test[i]).resize((s, s))
        img_cat.paste(img, (0, i * s))

    img_cat.save('images/'+ nettype + 'net_results/knn/'+ nettype + '_knn.jpg')

def render_tsne(checkpoint, classes):
    check = torch.load(checkpoint)
    net = ResNetLight(len(classes)).cuda()
    net.load_state_dict(check['model_state_dict'])
    net.eval()
    features = []

    def hook_feature(module, input, output):
        features.append(output)

    net._modules.get('avgpool').register_forward_hook(hook_feature)

    train_batch_size = 100
    data_root = './data'
    num_workers = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if 'classify' in checkpoint:
        nettype = 'classify'
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_test)
        trainloader = iter(torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers))
    else:
        nettype = 'rotate'
        trainloader = iter(torch.utils.data.DataLoader(Rot_Dataset(train=True), batch_size=train_batch_size, shuffle=True, num_workers=num_workers))

    X_train, y_train = next(trainloader)
    X_train = X_train.cuda()
    features = []
    net(X_train)
    feat = features[0].squeeze()
    feat = feat.detach().cpu().numpy()
    feat_proj = TSNE(n_components=2).fit_transform(feat)
    points = [[] for i in range(len(classes))]
    idx = y_train.detach().cpu().numpy()
    for i in range(100):
        points[idx[i]].append(feat_proj[i])
    points = [np.array(p).T for p in points]
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    for p in points:
        plt.scatter(p[0], p[1])
    plt.savefig('images/'+ nettype + 'net_results/tsne/'+ nettype +'_tsne .jpg')
