import torch.nn as nn
import torch
from torchsummary import summary


class MobileAlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False) :
        super(MobileAlexNet, self).__init__()
        self.num_classes = num_classes

        def conv_bn(inp, oup, kernel_size=11, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0) 
            )
 
        def conv_dw(inp, oup, kernel_size=3, padding=1,stride=1):
            return nn.Sequential(
                #输出通道与输入通道一样，groups=inp，进行分层卷积
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False), 
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_bn(3, 96, kernel_size=11, stride=4, padding=2) , # input[3, 224, 224]  output[96, 55, 55]
            conv_bn(96, 256, kernel_size=5, stride=4, padding=2) , # output[256, 13, 13]
            conv_dw(256,384, kernel_size=3, padding=1) , # output[384, 13, 13]
            conv_dw(384,384, kernel_size=3, padding=1) , # output[384, 13, 13]
            conv_dw(384, 256, kernel_size=3, padding=1) , # output[256, 13, 13]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2304, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
def mobilealexnet(num_classes): 
    model = MobileAlexNet(num_classes=num_classes)
    return model                



# net = AlexNet(num_classes=1000)
# summary(net.to('cuda'), (3,224,224))
#########################################################################################################################################
# Total params: 62,378,344
# Trainable params: 62,378,344
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 11.09
# Params size (MB): 237.95
# Estimated Total Size (MB): 249.62
# ----------------------------------------------------------------
# conv_parameters:  3,747,200
# fnn_parameters:  58,631,144   93% 的参数量
