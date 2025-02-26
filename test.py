import os
import glob
import argparse
import numpy as np
from PIL import Image
from random import sample

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from model import InpaintNet


def norm(x):
    return 2. * x - 1.  # [0,1] -> [-1,1]

def denorm(x):
    out = (x + 1) / 2  # [-1,1] -> [0,1]
    return out.clamp_(0, 1)

parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset', type=str, help='test dataset', default='HDTF', choices=['faceforensics', 'HDTF', 'VoxCeleb-ID'])
parser.add_argument('--model_name', default='checkpoints', type=str)
parser.add_argument('--model_pth', default='G_500000.pth', type=str)

def get_mask():

    mask = []
    IMAGE_SIZE = 256

    up = IMAGE_SIZE // 4 + 25
    down = IMAGE_SIZE - IMAGE_SIZE // 4 + 25
    left = IMAGE_SIZE // 4
    right = IMAGE_SIZE - IMAGE_SIZE // 4
        
    m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    m[up:down, left:right] = 1
    mask = np.array(m)
        
    return mask.astype(np.float32) #, np.array(points)

if __name__ == "__main__":
    args = parser.parse_args()
    args.checkpoint = os.path.join(args.model_name, args.model_pth)
    epoch = args.model_pth.split('.')[0].split('_')[1]

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    g_model = InpaintNet().to(device)
    print('# generator parameters:', sum(param.numel() for param in g_model.parameters()))
    
    print("load checkpoint:", args.checkpoint)
    g_checkpoint = torch.load(args.checkpoint, map_location=device)
    g_model.load_state_dict(g_checkpoint)
    g_model.eval()

    print("test_dataset:", args.test_dataset)

    to_tensor = transforms.ToTensor()

    if args.test_dataset == 'faceforensics':
        imgs_path = "Datasets/faceforensics/Images/Test/"
        voices_path = 'Datasets/faceforensics/Audios/Test/'

        samples = []
        samples_voice = []
        
        id_list = sorted(os.listdir(imgs_path))
        for id in id_list:
            img_files = sorted(glob.glob(os.path.join(imgs_path, id, '*.jpg')))
            aud_files = sorted(glob.glob(os.path.join(voices_path, id, '*.npy')))
            for img_file in img_files:
                samples.append(img_file)
                samples_voice.append(aud_files[0])


        assert len(samples) == len(samples_voice)

    elif args.test_dataset == 'HDTF':
        imgs_path = "Datasets/HDTF/Images/Test/"
        voices_path = 'Datasets/HDTF/Audios/Test/'

        samples = []
        samples_voice = []

        id_list = sorted(os.listdir(imgs_path))
        for id in id_list:
            img_files = sorted(glob.glob(os.path.join(imgs_path, id, '*.jpg')))
            aud_files = sorted(glob.glob(os.path.join(voices_path, id, '*.npy')))
            for img_file in img_files:
                samples.append(img_file)
                samples_voice.append(aud_files[0])

        assert len(samples) == len(samples_voice)

    elif args.test_dataset == 'VoxCeleb-ID':
        imgs_path = "Datasets/VoxCelebID/Images/Test/"
        voices_path = "Datasets/VoxCelebID/Audios/Test/"

        samples = []
        samples_voice = []

        id_list = sorted(os.listdir(imgs_path))
        for id in id_list:
            img_files = sorted(glob.glob(os.path.join(imgs_path, id, '*.jpg')))
            aud_files = sorted(glob.glob(os.path.join(imgs_path, id, '*.npy')))
            for img_file in img_files:
                samples.append(img_file)
                samples_voice.append(aud_files[0])

    print('Total number:', len(samples))

    # save_dir = f"test_{args.test_dataset}_{args.model_name}_{args.model_pth[:-4]}"
    save_dir = f"test_{args.test_dataset}_{args.model_name}_{epoch}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    import time
    start_time = time.time()
    
    for i in range(len(samples)):

        print(i, samples[i].split("/")[-1])
        mask = get_mask()
        mask = to_tensor(mask) 
        
        img = Image.open(samples[i]).convert('RGB').resize((256, 256))
        img = to_tensor(img)
        img = norm(img)  # [0,1] -> [-1,1]
        gt_img = img
        img = img * (1. - mask)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        
        fbank = np.squeeze(np.load(samples_voice[i]))
        audio = torch.from_numpy(fbank.astype('float32'))
        audio = audio.unsqueeze(dim=0)
                
        img = img.to(device)
        mask = mask.to(device)
        audio = audio.to(device)

        face, audioOut, face_id = g_model(img, mask, audio)

        id = os.path.basename(samples[i]).split('.')[0]
        save_image(denorm(gt_img), os.path.join(save_dir, id + "_gt.jpg"))
        save_image(denorm(img), os.path.join(save_dir, id + "_masked.jpg"))
        save_image(mask, os.path.join(save_dir, id + "_mask.jpg"))
        save_image(denorm(face), os.path.join(save_dir, id + "_final.jpg"))
        save_image(denorm(audioOut), os.path.join(save_dir, id + "_audioOut.jpg"))
        
    print("Done in %.3f seconds!" % (time.time() - start_time))
    
