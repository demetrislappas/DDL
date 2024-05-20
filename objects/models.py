import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# Layers

class ConvBlock2d(nn.Module):
    def __init__(self,in_dim,out_dim,kernels=[3,5,7],dropout=.0,decode=False,last_channel=False):
        super().__init__()
        k_in, k_out = (4, 1) if decode == True else (1, 4)
        self.last_channel = last_channel
        self.convs = nn.ModuleList([nn.Conv2d(in_dim,k_in*out_dim,kernel_size=kernel,padding='same') for kernel in kernels])
        self.batch_norm_1 = nn.BatchNorm2d(len(kernels)*k_in*out_dim)
        self.relu_1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.resize = Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)',p1=2,p2=2) if decode == True else Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w',p1=2,p2=2)
        self.conv_last = nn.Conv2d(len(kernels)*k_out*out_dim,out_dim,kernel_size=1,padding='same')
        self.batch_norm_2 = nn.BatchNorm2d(out_dim)
        self.relu_2 = nn.ReLU()
        
    
    def forward(self,x):
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x,dim=1)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.drop(x)
        x = self.resize(x) 
        x = self.conv_last(x)
        if self.last_channel == False:
            x = self.batch_norm_2(x)
            x = self.relu_2(x)
        return x

class EncoderBlock2d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,kernels=[3,5,7]):
        super().__init__()

        self.l3 = latent_dim//2
        self.l2 = self.l3//2
        self.l1 = self.l2//2
        
        self.conv_1 = ConvBlock2d(channels,self.l1,kernels)
        self.conv_2 = ConvBlock2d(self.l1,self.l2,kernels)
        self.conv_3 = ConvBlock2d(self.l2,self.l3,kernels)
        self.conv_4 = ConvBlock2d(self.l3,latent_dim,kernels)


    def forward(self,x):
       x1 = self.conv_1(x)
       x2 = self.conv_2(x1)
       x3 = self.conv_3(x2)
       x4 = self.conv_4(x3)
       return x1, x2, x3, x4

class SkipDecoderBlock2d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,kernels=[3,5,7]):
        super().__init__()

        self.l3 = latent_dim//2
        self.l2 = self.l3//2
        self.l1 = self.l2//2

        self.conv_1 = ConvBlock2d(latent_dim,self.l3,kernels,decode=True)
        self.conv_2 = ConvBlock2d(2*self.l3,self.l2,kernels,decode=True)
        self.conv_3 = ConvBlock2d(2*self.l2,self.l1,kernels,decode=True)
        self.conv_4 = ConvBlock2d(2*self.l1,channels,kernels,decode=True,last_channel=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x1, x2, x3, x4):

        x = self.conv_1(x4)
        x = torch.cat([x,x3],dim=1)
        x = self.conv_2(x)
        x = torch.cat([x,x2],dim=1)
        x = self.conv_3(x)
        x = torch.cat([x,x1],dim=1)
        x = self.conv_4(x) 
        return self.sigmoid(x)


# Models
    
class UNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=1):
        super().__init__()
        assert temporal==1, 'Temporal must be 1 for this model'

        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        self.mse_loss = nn.MSELoss()

    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(dim=2)
        x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4)
        return x

class Conv3dSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=3):
        super().__init__()
        self.target_frame = (temporal-1)//2
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        self.to_2d = Rearrange('b c t h w -> (b t) c h w')
        self.to_3d = Rearrange('(b t) c h w -> b c t h w',t=temporal)

        self.l3 = latent_dim//2
        self.l2 = self.l3//2
        self.l1 = self.l2//2

        kernel_sizes = [3,3,3,3]
        self.conv1 = nn.Conv3d(self.l1,self.l1,kernel_size=(temporal,kernel_sizes[0],kernel_sizes[0]),padding='same')
        self.conv2 = nn.Conv3d(self.l2,self.l2,kernel_size=(temporal,kernel_sizes[1],kernel_sizes[1]),padding='same')
        self.conv3 = nn.Conv3d(self.l3,self.l3,kernel_size=(temporal,kernel_sizes[2],kernel_sizes[2]),padding='same')
        self.conv4 = nn.Conv3d(latent_dim,latent_dim,kernel_size=(temporal,kernel_sizes[3],kernel_sizes[3]),padding='same')

        
    def forward(self,x):
        # Stack temporal onto batch
        x = self.to_2d(x)
        
        # Pass through 2D encoder
        x1, x2, x3, x4 = self.encoder(x)
        
        # Convert each skip connection to 3D
        x1 = self.to_3d(x1)
        x2 = self.to_3d(x2)
        x3 = self.to_3d(x3)
        x4 = self.to_3d(x4)
        
        # Pass skip connections through masked convolutions
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        
        # Convert each skip connection back to 2D
        x1 = x1[:,:,self.target_frame]
        x2 = x2[:,:,self.target_frame]
        x3 = x3[:,:,self.target_frame]
        x4 = x4[:,:,self.target_frame]
        
        # Pass through 2D decoder
        x = self.decoder(x1, x2, x3, x4)
        
        return x   
