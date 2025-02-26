import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import pretrained_models.VGGFace2.models.resnet as ResNet


N_IDENTITY = 8631
 
def denorm(x):
    out = (x + 1) / 2 # [-1,1] -> [0,1]
    return out.clamp_(0, 1)

class VGGFaceIDLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
        resume_path = 'pretrained_models/VGGFace2/resnet50_ft_weight.pkl'
        self.model = ResNet.resnet50(weights_path=resume_path, num_classes = N_IDENTITY, include_top = False)
        # checkpoint = torch.load(resume_path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Resume from {resume_path}")

    def forward(self, es, gt):
        with torch.no_grad():
            es_id = self.model(es)        
            gt_id = self.model(gt)
        loss = torch.abs(es_id - gt_id).mean()

        return loss

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        self.enc_4 = nn.Sequential(*vgg16.features[17:23])

        #print(self.enc_1)
        #print(self.enc_2)
        #print(self.enc_3)
        #print(self.enc_4)

        # fix the encoder
        for i in range(4):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    

class VGG16Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.vgg = VGG16FeatureExtractor()
        self.l2 = nn.MSELoss()

    def forward(self, img1, img2, mask):
        # https://pytorch.org/docs/stable/torchvision/models.html
        # Pre-trained VGG16 model expect input images normalized in the same way.
        # The images have to be loaded in to a range of [0, 1]
        # and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        img1 = img1 * mask
        img1 = denorm(img1) # [-1,1] -> [0,1]
        img1 = self.normalize(img1[0]) # BxCxHxW -> CxHxW -> normalize
        img1 = img1.unsqueeze(0) # CxHxW -> BxCxHxW
        vgg_img1 = self.vgg(img1)[-1]

        img2 = img2 * mask
        img2 = denorm(img2) # [-1,1] -> [0,1]
        img2 = self.normalize(img2[0]) # BxCxHxW -> CxHxW -> normalize
        img2 = img2.unsqueeze(0) # CxHxW -> BxCxHxW
        vgg_img2 = self.vgg(img2)[-1]

        lossvalue = self.l2(vgg_img1, vgg_img2) 
        return lossvalue


def calc_gan_loss(discriminator, output, target):
    y_pred_fake = discriminator(output, target)
    y_pred = discriminator(target, output)

    g_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) + 1.) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - 1.) ** 2))/2
    d_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) - 1.) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + 1.) ** 2))/2

    return g_loss, d_loss
