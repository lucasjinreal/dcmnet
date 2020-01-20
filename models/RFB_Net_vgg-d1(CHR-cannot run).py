import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False,up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0: #2 forward up_size
            x = F.upsample(input=x, size=(self.up_size, self.up_size), mode='bilinear')
        return x

class BasicConvBN(nn.Module): # +

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=True):
        super(BasicConvBN, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )#torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中，并且使用torch.nn.Sequential会自动加入激励函数
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out



class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras,pyramid_ext,head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        #self.x1 = x1
        #self.num_classes = num_classes
        self.size = size
        self.pyramid_ext = nn.ModuleList(pyramid_ext) # +

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return

        # vgg network
        #self.base = nn.ModuleList(base)
        self.base = base
        # conv_4
        self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        self.extras = nn.ModuleList(extras)
        #self.ft_module = nn.ModuleList(ft_module)#5 定义一个列表名为ft_module，作为中间参数传递用

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
        #以下各层分别定义extra_layers0-6层，其中只取0（RFB模块，输入/输出1024),1(10*10RFBNET stride2 1024/512）,2（5*5RFBNET stride2 512/256）,4（3*3conv10_2 256/256),6(1*1conv11_2 256/256)使用
        self.BasicRFB1 = BasicRFB(1024, 1024, scale=1.0, visual=2)
        self.BasicRFB2 = BasicRFB(1024, 512, stride=2, scale=1.0, visual=2)
        self.BasicRFB3 = BasicRFB(512, 256, stride=2, scale=1.0, visual=1)
        self.BasicConva1 = BasicConv(256, 128, kernel_size=1, stride=1)
        self.BasicConva2 = BasicConv(128, 256, kernel_size=3, stride=1)
        self.BasicConva3 = BasicConv(256, 128, kernel_size=1, stride=1)
        self.BasicConva4 = BasicConv(128, 256, kernel_size=3, stride=1)

    def _upsample_add(self,x,y): #3 定义上采样

        _, _, H, W=y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear')
        return  torch.cat([z,y],1)

    # 作用1：通过调用BasicConv，并且传入size值，将19*19和10*10大小的特征图变为38*38（300*300）或64*64（512*512）
    # 作用2：对要进行拼接的三个特征图（38*38，19*19，10*10）进行降维    #why?


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        transformed_features = list()#6 定义一个列表名为transformed_features作为中间特征传递
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.Norm(x)
        sources.append(s) #sources.append(0)是conv4_3 relu的特征图

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        sources.append(x) #4 输出fc7层特征
                            #sources.append(1)是fc7的特征图
        #print(len(self.base))
        # apply extra layers and cache source layer outputs
        '''
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 ==0:
                sources.append(x)
        '''
        #extras
        #x = BasicRFB(1024, 1024, scale=1.0, visual=2)(x) #0
        s1 = self.BasicRFB1(x)
        sources.append(s1)  #sources.append(2)是trick-1 RFB的特征图
        #x = BasicRFB(1024, 512, stride=2, scale=1.0, visual=2)(x)#1
        s2 = self.BasicRFB2(s1)
        sources.append(s2)  #sources.append(3)是第三层网络RFB stride2 10*10*512的特征图
        #x = BasicRFB(512, 256, stride=2, scale=1.0, visual=1)(x)#2
        s3 = self.BasicRFB3(s2)
        sources.append(s3)  #sources.append(4)是第四层网络RFB stride2 5*5*256的特征图
        #x = BasicConv(256, 128, kernel_size=1, stride=1)(x)#第3层不输出
        s4 = self.BasicConva1(s3)
        #x = BasicConv(128, 256, kernel_size=3, stride=1)(x)#4
        s5 = self.BasicConva2(s4)
        sources.append(s5)   #sources.append(5)是第五层网络conv10_2 3*3*256的特征图
        #x = BasicConv(256, 128, kernel_size=1, stride=1)(x)#第5层不输出
        s6 = self.BasicConva3(s5)
        #x = BasicConv(128, 256, kernel_size=3, stride=1)(x)#6
        s7 = self.BasicConva4(s6)
        sources.append(s7)   #sources.append(6)是第六层网络conv11_2 1*1*256的特征图
        x = s7
        #print(len(sources)) #测试传入sources的共有几层网络
        for k, v in enumerate(self.sources): #用enumerate函数k,v键值对将sources列表里存的值取出调用
            transformed_features.append(v(sources[k])) #以v(sources[k]的方式传入中间列表transformed_features
        x = torch.cat(transformed_features, 1) # torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接
                                              #按维数1拼接就是：这此处，当k取不同值时，将不同v(sources[k])横着拼接起来
        pyramid_fea = list() #定义两个空列表用来装上采样之后的各层特征图
        pyramid_fea1 = list() #pyramid_fea差不多跟sources各层对应，pyramid_fea1对应新生成的各层特征图
        for k, v in enumerate(self.pyramid_ext): #将新定义的特征层pyramid_ext按键-值对传给pyramid_fea
            x = v(x)
            pyramid_fea.append(x)
        # print(pyramid_fea[2])
        l4_3 = self._upsample_add(pyramid_fea[1],pyramid_fea[0])  # pyramid_fea[0]（256*3,512），pyramid_fea[1]（512,512）
        l4_3_1 = self.conv1(l4_3)
        l4_3_11 = self.bn1(l4_3_1)
        l4_3_111 = F.relu(l4_3_11)
        pyramid_fea1.append(l4_3_111)  # pyramid_fea1用来存最新的特征图层

        l7_2 = self._upsample_add(pyramid_fea[2], pyramid_fea[1])
        l7_2_1 = self.conv2(l7_2)
        l7_2_11 = self.bn2(l7_2_1)
        l7_2_111 = F.relu(l7_2_11)
        pyramid_fea1.append(l7_2_111)

        l8_2 = self._upsample_add(pyramid_fea[3], pyramid_fea[2])
        l8_2_1 = self.conv3(l8_2)
        l8_2_11 = self.bn3(l8_2_1)
        l8_2_111 = F.relu(l8_2_11)
        pyramid_fea1.append(l8_2_111)

        l9_2 = self._upsample_add(pyramid_fea[4], pyramid_fea[3])
        l9_2_1 = self.conv4(l9_2)
        l9_2_11 = self.bn4(l9_2_1)
        l9_2_111 = F.relu(l9_2_11)
        pyramid_fea1.append(l9_2_111)

        l10_2 = self._upsample_add(pyramid_fea[5], pyramid_fea[4])
        l10_2_1 = self.conv5(l10_2)
        l10_2_11 = self.bn5(l10_2_1)
        l10_2_111 = F.relu(l10_2_11)
        pyramid_fea1.append(l10_2_111)

        l11_2_1 = self.conv6(pyramid_fea[5])
        l11_2_11 = F.relu(l11_2_1)
        pyramid_fea1.append(l11_2_11)

        # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W]
        # apply multibox head to source layers
        for (x, l, c) in zip(pyramid_fea1, self.loc, self.conf):
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W),
            # 因此, 这里的排列是将其改为 (N, H, W, C).
            # contiguous返回内存连续的tensor, 由于在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的,
            # 然后 view 操作是基于连续地址的, 因此, 需要调用contiguous语句.
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # loc: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]???
        # conf: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes
        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 最终的shape为(两维):[batch, num_boxes*num_classes]
        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4), # [batch, num_boxes, 4], [1, 8732, 4]                  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):#区分rfbnet里面的base是vgg16，而ssd_resnet101里面的base是resnet101
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            # Load all tensors onto the CPU, using a function
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

'''
def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=1)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2)]
            else:
                layers += [BasicRFB(in_channels, v, scale = 1.0, visual=2)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    elif size ==300:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers
'''

def add_extras(): #纯手工定义各层extras
    layers = []
    layers += [BasicRFB(1024, 1024, scale=1.0, visual=2)] #tricks之一RFB
    layers += [BasicRFB(1024, 512, stride=2, scale=1.0, visual=2)] #第三层10*10*512
    layers += [BasicRFB(512, 256,stride=2, scale=1.0, visual=1)] #第四层5×5*256
    layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
    layers += [BasicConv(128, 256, kernel_size=3, stride=1)] #第五层3×3*256
    layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
    layers += [BasicConv(128, 256, kernel_size=3, stride=1)] #第六层1*1*256
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256]
}

#作用1：通过调用BasicConv，并且传入size值，将19*19和10*10大小的特征图变为38*38（300*300）
#作用2：对要进行拼接的三个特征图（38*38，19*19，10*10）进行降维    #这个是上采样操作这样才能进行后面的concate
def feature_transform_module(add_extras, size):  # +
    if size == 300:
        up_size = 38
    elif size == 512:
        up_size = 64

    layers = []
    # conv4_3
    # layers += [BasicConv(1024, 256, kernel_size=1, padding=0)]
    layers += [BasicConv(512, 256, kernel_size=1, padding=0)]
    # fc_7
    # layers += [BasicConv(2048, 256, kernel_size=1, padding=0, up_size=up_size)]
    layers += [BasicConv(1024, 256, kernel_size=1, padding=0, up_size=up_size)]

    layers += [BasicConv(add_extras[1].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]
    return add_extras, layers


# 合并后生成新特征图，经过layers中的卷积后，特征图的尺寸依次为：38*38，19*19，10*10，5*5，3*3，1*1
def pyramid_feature_extractor(size):  # + 类似FPN里面的latlayers侧面层，在上采样操作前对自下向上提取的原始特征进行一个初步操作
    if size == 300:
        layers = [BasicConvBN(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConvBN(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConvBN(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConvBN(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConvBN(256, 256, kernel_size=3, stride=1, padding=0),
                  BasicConvBN(256, 256, kernel_size=3, stride=1, padding=0, bn=False)]
    elif size == 512:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers

def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    #return RFBNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
     #                           add_extras(size, extras[str(size)], 1024),
      #                          mbox[str(size)], num_classes), num_classes)
    return RFBNet(phase, size,num_classes, *multibox(size, vgg(base[str(size)], 3),add_extras(),mbox[str(size)], num_classes),num_classes)
                  #*feature_transform_module(add_extras(extras[str(size)], 1024), size=size))
                                #pyramid_ext=pyramid_feature_extractor(size))
#phase, size, base, extras,pyramid_ext,head, num_classes