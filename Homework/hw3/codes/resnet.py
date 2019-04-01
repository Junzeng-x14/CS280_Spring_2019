import torch.nn as nn

def conv2d(in_C, out_C, k, s = 1, p = 1):
    return nn.Conv2d(in_C, out_C, kernel_size=k, stride=s, padding=p, bias=False)

class ResnetBlock(nn.Module):
    def __init__(self, in_C, out_C, s = 1, downsample = None):
        super(ResnetBlock, self).__init__()

        self.conv = nn.Sequential(
            conv2d(in_C, out_C, k = 3, s = s),
            nn.BatchNorm2d(out_C),
            nn.ReLU(),
            conv2d(out_C, out_C, k = 3),
            nn.BatchNorm2d(out_C)
        )

        if downsample:
            self.downsample = downsample
        else:
            self.downsample = lambda x : x

    def forward(self, x):
        return  nn.functional.relu(self.downsample(x) + self.conv(x))


class ResNetLight(nn.Module):
    layers = [1, 1, 1, 1]
    in_C = 32

    def __init__(self, num_classes=10, dropout = True):
        super(ResNetLight, self).__init__()
        self.start = nn.Sequential(
            conv2d(3, self.in_C, k = 7, s = 1, p = 3),
            nn.BatchNorm2d(self.in_C),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.resblocks1 = self.resblocks(self.in_C, self.layers[0])
        self.resblocks2 = self.resblocks(32*2, self.layers[1], s = 1)
        self.resblocks3 = self.resblocks(32*4, self.layers[2], s = 1)
        self.resblocks4 = self.resblocks(32*8, self.layers[3], s = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout:
            self.dropout = nn.Dropout()
        else:
            self.dropout = lambda x: x
        self.fc = nn.Linear(32*8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def resblocks(self, out_C, num_blocks, s = 1):
        downsample = None
        if s != 1 or self.in_C != out_C:
            downsample = nn.Sequential(
                conv2d(self.in_C, out_C, k = 1, s = s, p = 0),
                nn.BatchNorm2d(out_C)
            )

        layers = []
        layers.append(ResnetBlock(self.in_C, out_C, s, downsample))
        self.in_C = out_C
        for _ in range(1, num_blocks):
            layers.append(ResnetBlock(self.in_C, out_C))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.start(x)
        x = self.resblocks1(x)
        x = self.resblocks2(x)
        x = self.resblocks3(x)
        x = self.resblocks4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
