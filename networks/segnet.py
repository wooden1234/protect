import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self,ch_in, ch_out):
        super(SegNet,self).__init__()
        self.n_channels = ch_in
        self.n_classes  = ch_out
        self.encode_Conv1 = nn.Sequential(
            nn.Conv2d(ch_in,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encode_Conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.encode_Conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.encode_Conv4 = nn.Sequential(
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.encode_Conv5 = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.decode_Conv1 = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decode_Conv2 = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decode_Conv3 = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decode_Conv4 = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decode_Conv5 = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, ch_out, 3, padding=1),
            nn.Sigmoid()
        )
        self.weights_new = self.state_dict()

    def forward(self, x):
        #print('inputsize',x.shape)
        x = self.encode_Conv1(x)
        x1size = x.size()
        x,id1 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        #print(x.shape,id1.shape)
        x = self.encode_Conv2(x)
        x2size = x.size()
        x,id2 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        #print(x.shape,id2.shape)
        x = self.encode_Conv3(x)
        x3size = x.size()
        x,id3 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        #print(x.shape,id3.shape)
        x = self.encode_Conv4(x)
        x4size = x.size()
        x,id4 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        #print(x.shape,id4.shape)
        x = self.encode_Conv5(x)
        x5size = x.size()
        x,id5 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        #print(x.shape,id5.shape)
        x = F.max_unpool2d(x,indices=id5,kernel_size=2, stride=2,output_size=x5size)
        x = self.decode_Conv1(x)
        #print(x.shape,id4.shape)
        x = F.max_unpool2d(x,indices=id4,kernel_size=2, stride=2,output_size=x4size)
        x = self.decode_Conv2(x)
        x = F.max_unpool2d(x,indices=id3,kernel_size=2, stride=2,output_size=x3size)
        x = self.decode_Conv3(x)
        x = F.max_unpool2d(x,indices=id2,kernel_size=2,stride=2,output_size=x2size)
        x = self.decode_Conv4(x)
        x = F.max_unpool2d(x,indices=id1,kernel_size=2,stride=2,output_size=x1size)
        x = self.decode_Conv5(x)
        return x

if __name__=='__main__':
    model = SegNet(3,7)
    print(model)
    from thop import profile
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
    out = model(input)
    print(out.shape)