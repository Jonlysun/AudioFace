
import os
import tqdm
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from data import DS, InfiniteSampler
from loss import VGG16Loss, calc_gan_loss, VGGFaceIDLoss
from model import InpaintNet, PatchDiscriminator

def Train(args):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True


    size = (args.image_size, args.image_size)
    train_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ])

    train_set = DS(train_tf)
    iterator_train = iter(data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=InfiniteSampler(len(train_set)),
        num_workers=args.n_threads))
    print(len(train_set))

    g_model = InpaintNet(audio_grad=args.audio_grad).to(device)
    pd_model = PatchDiscriminator().to(device)
    l1 = nn.L1Loss().to(device)
    vggloss = VGG16Loss().to(device)

    start_iter = 0

    g_optimizer = torch.optim.Adam(
        g_model.parameters(), 
        args.lr, (args.b1, args.b2))

    pd_optimizer = torch.optim.Adam(
        pd_model.parameters(),
        args.lr, (args.b1, args.b2))

    if args.resume:
        g_checkpoint = torch.load(f'./snapshots_FFHQ_AFF/ckpt/G_{args.resume}.pth', map_location=device)
        g_model.load_state_dict(g_checkpoint)

        # g_checkpoint = torch.load(f'./pretrained_models/AFF_FFHQ/G_{args.resume}.pth', map_location=device)
        # audiofeat_g_checkpoint = {}
        # coarse_g_checkpoint = {}
        # for key in g_model.state_dict().keys():
        #     if key in g_checkpoint.keys():            
        #         if "audioFeat" in key:
        #             save_key = '.'.join(key.split('.')[1:])
        #             audiofeat_g_checkpoint[save_key] = g_checkpoint[key]
        #         if "coarse" in key:
        #             save_key = '.'.join(key.split('.')[1:])
        #             coarse_g_checkpoint[save_key] = g_checkpoint[key]            
        # g_model.audioFeat.load_state_dict(audiofeat_g_checkpoint) 
        # g_model.coarse.load_state_dict(coarse_g_checkpoint) 

        pd_checkpoint = torch.load(f'./snapshots_FFHQ_AFF/ckpt/PD_{args.resume}.pth', map_location=device)
        pd_model.load_state_dict(pd_checkpoint)

        g_model.audioFeat.Enet.load_state_dict(torch.load('pretrained_models/Voice2Face/voice_embedding.pth'))
        g_model.audioFeat.Gnet.load_state_dict(torch.load('pretrained_models/Voice2Face/generator.pth'))
        print('Models restored')

    faceidloss_fn = VGGFaceIDLoss().to(device)

    for i in tqdm.tqdm(range(start_iter, args.max_iter)):

        img, mask, vox, face_id_gt = [x.to(device) for x in next(iterator_train)]
        img = 2. * img - 1. # [0,1] -> [-1,1]
        masked = img * (1. - mask)

        coarse_result, audioOut, face_id_pred = g_model(masked, mask, vox)

        pg_loss, pd_loss = calc_gan_loss(pd_model, coarse_result, img)

        face_id_loss = l1(face_id_pred, face_id_gt)
        recon_loss = l1(coarse_result, img) + 0.01*l1(audioOut, img)
        faceidloss = faceidloss_fn(coarse_result, audioOut)

        gan_loss = pg_loss 
        total_loss = 1*recon_loss + 0.002*gan_loss + 0.001*face_id_loss + faceidloss

        g_optimizer.zero_grad()
        pd_optimizer.zero_grad()

        total_loss.backward(retain_graph=True)
        pd_loss.backward()

        g_optimizer.step()    
        pd_optimizer.step()

        if (i) % args.save_model_interval == 0 or (i) == args.max_iter:
            torch.save(g_model.state_dict(), f'{args.save_dir}/G_{i}.pth')
            torch.save(pd_model.state_dict(), f'{args.save_dir}/PD_{i}.pth')

        if (i) % args.log_interval == 0:

            print('\n', i)
                
            print('g_loss/total_loss', total_loss.item())
            print('g_loss/recon_loss', recon_loss.item())
            print('g_loss/vgg_loss', faceidloss.item())
            print('d_loss/pd_loss', pd_loss.item())

        def denorm(x):
            out = (x + 1) / 2 # [-1,1] -> [0,1]
            return out.clamp_(0, 1)
        if (i) % args.vis_interval == 0: 
            ims = torch.cat([img, masked, coarse_result, audioOut], dim=3)

            ims_train = ims.add(1).div(2).mul(255).clamp(0, 255).byte()
            ims_train = ims_train[0].permute(1, 2, 0).data.cpu().numpy()
            
            ims_out = Image.fromarray(ims_train)   
            fullpath = '%s/iteration%d.png' % (args.training_result, i)
            ims_out.save(fullpath)
            
            print("train image saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./save_ckpts/')
    parser.add_argument('--training_result', type=str, default='./training_result/')
    parser.add_argument('--resume', type=int, default=0)

    parser.add_argument('--audio_grad', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--max_iter', type=int, default=600000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_threads', type=int, default=4)
    parser.add_argument('--save_model_interval', type=int, default=50000)
    parser.add_argument('--vis_interval', type=int, default=100000)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.training_result, exist_ok=True)
    print('save_dir: ', args.save_dir)
    print('training_result: ', args.training_result)
    print('resume: ', args.resume)
    print('audio_grad: ', args.audio_grad)
    print('lr: ', args.lr)
    print('b1: ', args.b1)
    print('b2: ', args.b2)
    print('max_iter: ', args.max_iter)
    print('batch_size: ', args.batch_size)
    print('n_threads: ', args.n_threads)
    print('save_model_interval: ', args.save_model_interval)


    Train(args)