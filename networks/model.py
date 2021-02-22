import torch.nn as nn



class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv1d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2,inplace=False))
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.head=ConvBlock(1,16,ker_size=3,padd=1,stride=1)
        self.body=ConvBlock(16,16,ker_size=3,padd=1,stride=1)
        self.tail=nn.Conv1d(16 ,1,kernel_size=3,stride=1,padding=1)
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x+y

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.head=ConvBlock(1,16,ker_size=3,padd=1,stride=1)
        self.body=ConvBlock(16,16,ker_size=3,padd=1,stride=1)
        self.tail=nn.Conv1d(16 ,1,kernel_size=3,stride=1,padding=1)
        self.tanh=nn.Tanh()
    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tanh(self.tail(x))
        return x
    
def init_models(device='cpu'):
    netG=Generator().to(device)
    netD=Discriminator().to(device)
    return netD,netG