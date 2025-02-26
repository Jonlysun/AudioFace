import torch
from torch import nn
import torch.nn.functional as F
from fusion import AVFFAttention


def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_act(name):
    if name == 'relu':
        activation = nn.ReLU(inplace=True)
    elif name == 'elu':
        activation == nn.ELU(inplace=True)
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


class CoarseEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class CoarseDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super().__init__()
        
        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class CoarseNet(nn.Module):
    def __init__(self, c_img=3, norm='instance', act_en='leaky_relu', act_de='relu'):
        super().__init__()

        cnum = 64

        self.en_1 = nn.Conv2d(c_img, cnum, 4, 2, padding=1)
        self.en_2 = CoarseEncodeBlock(cnum, cnum*2, 4, 2, normalization=norm, activation=act_en)
        self.en_3 = CoarseEncodeBlock(cnum*2, cnum*4, 4, 2, normalization=norm, activation=act_en)
        self.en_4 = CoarseEncodeBlock(cnum*4, cnum*8, 4, 2, normalization=norm, activation=act_en)
        self.en_5 = CoarseEncodeBlock(cnum*8, cnum*8, 4, 2, normalization=norm, activation=act_en)
        self.en_6 = CoarseEncodeBlock(cnum*8, cnum*8, 4, 2, normalization=norm, activation=act_en)
        self.en_7 = CoarseEncodeBlock(cnum*8, cnum*8, 4, 2, normalization=norm, activation=act_en)
        self.en_8 = CoarseEncodeBlock(cnum*8, cnum*8, 4, 2, activation=act_en)

        self.linear_model = nn.Sequential(nn.Linear(cnum*9, cnum*8), nn.ReLU(inplace=True))

        self.de_8 = CoarseDecodeBlock(cnum*8, cnum*8, 4, 2, normalization=norm, activation=act_de)
        self.de_7 = CoarseDecodeBlock(cnum*8*2, cnum*8, 4, 2, normalization=norm, activation=act_de)
        self.de_6 = CoarseDecodeBlock(cnum*8*2, cnum*8, 4, 2, normalization=norm, activation=act_de)
        self.de_5 = CoarseDecodeBlock(cnum*8*2, cnum*8, 4, 2, normalization=norm, activation=act_de)
        self.de_4 = CoarseDecodeBlock(cnum*8*2+256, cnum*4, 4, 2, normalization=norm, activation=act_de)
        self.de_3 = CoarseDecodeBlock(cnum*4*2+128, cnum*2, 4, 2, normalization=norm, activation=act_de)
        self.de_2 = CoarseDecodeBlock(cnum*2*2+64, cnum, 4, 2, normalization=norm, activation=act_de)
        self.de_1 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum*2+32, c_img, 4, 2, padding=1),
            get_act('tanh'))

        self.fusion1 = AVFFAttention(channels=256)
        self.fusion2 = AVFFAttention(channels=128)
        self.fusion3 = AVFFAttention(channels=64)
        self.fusion4 = AVFFAttention(channels=32)
    
    def forward(self, x, audio_emd, x16, x32, x64, x128):
        out_1 = self.en_1(x)
        out_2 = self.en_2(out_1)
        out_3 = self.en_3(out_2) # torch.Size([4, 256, 32, 32])
        out_4 = self.en_4(out_3) # torch.Size([4, 512, 16, 16])
        out_5 = self.en_5(out_4) # torch.Size([4, 512, 8, 8])
        out_6 = self.en_6(out_5) # torch.Size([4, 512, 4, 4])
        out_7 = self.en_7(out_6)
        out_8 = self.en_8(out_7) # torch.Size([4, 512, 1, 1])

        face_id = out_8.clone().view(out_8.shape[0], -1)
        audio_emd = audio_emd.view(audio_emd.size()[0], -1)        
        mid_feat = torch.cat([face_id, audio_emd], dim=1)  # [bs, 512+64]
        mid_feat = self.linear_model(mid_feat)  # [bs, 512+64]
        out_8 = mid_feat.clone().view(mid_feat.shape[0], -1, 1, 1)

        dout_8 = self.de_8(out_8)
        dout_8_out_7 = torch.cat([dout_8, out_7], 1)
        dout_7 = self.de_7(dout_8_out_7) # torch.Size([4, 512, 4, 4])
        dout_7_out_6 = torch.cat([dout_7, out_6], 1)
        dout_6 = self.de_6(dout_7_out_6) # torch.Size([4, 512, 8, 8])
        dout_6_out_5 = torch.cat([dout_6, out_5], 1)
        dout_5 = self.de_5(dout_6_out_5) # torch.Size([4, 512, 16, 16])
        fusion_1 = self.fusion1(dout_5, x16)

        dout_5_out_4 = torch.cat([dout_5, out_4, fusion_1], 1)
        # dout_5_out_4 = torch.cat([dout_5, fusion_1, fusion_1, fusion_1], 1)

        # dout_5_out_4 = torch.cat([dout_5, out_4, x16], 1)
        dout_4 = self.de_4(dout_5_out_4) # torch.Size([4, 256, 32, 32])
        fusion_2 = self.fusion2(dout_4, x32)
        dout_4_out_3 = torch.cat([dout_4, out_3, fusion_2], 1)
        # dout_4_out_3 = torch.cat([dout_4, out_3, x32], 1)
        dout_3 = self.de_3(dout_4_out_3) # torch.Size([4, 128, 64, 64])
        fusion_3 = self.fusion3(dout_3, x64)
        dout_3_out_2 = torch.cat([dout_3, out_2, fusion_3], 1)
        # dout_3_out_2 = torch.cat([dout_3, out_2, x64], 1)
        dout_2 = self.de_2(dout_3_out_2) # torch.Size([4, 64, 128, 128])
        fusion_4 = self.fusion4(dout_2, x128)
        dout_2_out_1 = torch.cat([dout_2, out_1, fusion_4], 1)
        dout_1 = self.de_1(dout_2_out_1) # torch.Size([4, 3, 256, 256])

        return dout_1, mid_feat

class VoiceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(VoiceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x

class Generator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True), # torch.Size([4, 1024, 4, 4])
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True), # torch.Size([4, 512, 8, 8])
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True), # torch.Size([4, 256, 16, 16])
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True), # torch.Size([4, 128, 32, 32])
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True), # torch.Size([4, 64, 64, 64])
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[4], output_channel, 1, 1, 0, bias=True), # torch.Size([4, 3, 64, 64])
        )

    def forward(self, x):
        # x = self.model(x)       

        for i in range(len(self.model)):
            x = self.model[i](x)
            if i == 4:
                x16 = x
            if i == 6:
                x32 = x
            if i == 8:
                x64 = x
            if i == 9:
                x64R = x
            # if i == 10:
            #     out = x           

        return x16, x32, x64, x64R

class AudioNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.Enet = VoiceEmbedNet(64, [256, 384, 576, 864], 64)        
        self.Enet.load_state_dict(torch.load('pretrained_models/Voice2Face/voice_embedding.pth'))
        self.Enet.eval()

        self.Gnet = Generator(64, [1024, 512, 256, 128, 64], 3)
        self.Gnet.load_state_dict(torch.load('pretrained_models/Voice2Face/generator.pth'))   
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=True), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 1, 1, 0, bias=True),
        )

    def forward(self, x):

        with torch.no_grad():
            embedding = self.Enet(x)
            embedding = F.normalize(embedding) # torch.Size([4, 64, 1, 1])
        x16, x32, x64, x64R = self.Gnet(embedding)     

        x = x64R
        for i in range(len(self.upsample)):
            x = self.upsample[i](x)
            if i == 0:
                x128 = x
            # if i == 2:
            #     x256 = x
            if i == 4:
                audioout = x  

        return embedding, x16, x32, x64, x128, audioout

class InpaintNet(nn.Module):
    def __init__(self, audio_grad=True):
        super().__init__()        
        self.audio_grad = audio_grad

        self.coarse = CoarseNet()
        self.audioFeat = AudioNet()

        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
        )

    def forward(self, image, mask, audio):

        embedding, x16, x32, x64, x128, audioout = self.audioFeat(audio)    

        out_c, mid_feat = self.coarse(image, embedding, x16, x32, x64, x128)
        out_c = image * (1. - mask) + out_c * mask

        face_id = self.mlp(mid_feat)

        return out_c, audioout, face_id


class PatchDiscriminator(nn.Module):
    def __init__(self, c_img=3,
                 norm='instance', act='leaky_relu'):
        super().__init__()

        c_in = c_img + c_img
        cnum = 64
        self.discriminator = nn.Sequential(
            nn.Conv2d(c_in, cnum, 4, 2, 1),
            get_act(act),

            nn.Conv2d(cnum, cnum*2, 4, 2, 1),
            get_norm(norm, cnum*2),
            get_act(act),

            nn.Conv2d(cnum*2, cnum*4, 4, 2, 1),
            get_norm(norm, cnum*4),
            get_act(act),

            nn.Conv2d(cnum*4, cnum*8, 4, 1, 1),
            get_norm(norm, cnum*8),
            get_act(act),

            nn.Conv2d(cnum*8, 1, 4, 1, 1))
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        return self.discriminator(x)

