import os
import glob
import argparse
from PIL import Image
from random import sample

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

from model import InpaintNet

def norm(x):
    return 2. * x - 1.  # [0,1] -> [-1,1]


def denorm(x):
    out = (x + 1) / 2  # [-1,1] -> [0,1]
    return out.clamp_(0, 1)


parser = argparse.ArgumentParser()
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

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    g_model = InpaintNet().to(device)
    print("load checkpoint:", args.checkpoint)
    g_checkpoint = torch.load(args.checkpoint, map_location=device)
    g_model.load_state_dict(g_checkpoint)
    g_model.eval()

    to_tensor = transforms.ToTensor()

    id_1_name = 'MEAD_M003'
    face_imgs_1 = sorted(glob.glob(os.path.join('test_data', 'img', id_1_name, f'*.jpg')))
    face_audio_1 = sorted(glob.glob(os.path.join('test_data', 'melspect', id_1_name, f'*.npy')))
    
    id_2_name = 'MEAD_M005'
    face_imgs_2 = sorted(glob.glob(os.path.join('test_data', 'img', id_2_name, f'*.jpg')))
    face_audio_2 = sorted(glob.glob(os.path.join('test_data', 'melspect', id_2_name, f'*.npy')))

    id_3_name = 'MEAD_W011'
    face_imgs_3 = sorted(glob.glob(os.path.join('test_data', 'img', id_3_name, f'*.jpg')))
    face_audio_3 = sorted(glob.glob(os.path.join('test_data', 'melspect', id_3_name, f'*.npy')))

    id_4_name = 'MEAD_W014'
    face_imgs_4 = sorted(glob.glob(os.path.join('test_data', 'img', id_4_name, f'*.jpg')))
    face_audio_4 = sorted(glob.glob(os.path.join('test_data', 'melspect', id_4_name, f'*.npy')))

    import time
    start_time = time.time()

    # img_path = face_imgs_1[0]
    # audio_path_list = face_audio_2

    img_path = face_imgs_2[0]
    audio_path_list = face_audio_4
    
    save_dir = f"test_data/results_ownaudio/{id_2_name}_Audio_{id_4_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for audio_path in audio_path_list:
        print(audio_path)

        mask = get_mask()        
        mask = to_tensor(mask)

        img = Image.open(img_path).convert('RGB').resize((256, 256))
        img = to_tensor(img)
        img = norm(img)  # [0,1] -> [-1,1]
        gt_img = img
        img = img * (1. - mask)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        
        fbank = np.squeeze(np.load(audio_path))
        audio = torch.from_numpy(fbank.astype('float32'))
        audio = audio.unsqueeze(dim=0)

        img = img.to(device)
        mask = mask.to(device)
        audio = audio.to(device)

        # audio = torch.zeros_like(audio).to(device)

        # face, audio_result, coarse_result = g_model(img, mask, audio)
        with torch.no_grad():
            face, audioOut, face_id = g_model(img, mask, audio)

        # audio_result = F.interpolate(audio_result, scale_factor=4.0, align_corners=True, mode='bilinear')

        id = os.path.basename(audio_path).split('.')[0]
        save_image(denorm(gt_img), os.path.join(save_dir, id + "_gt.jpg"))
        save_image(denorm(img), os.path.join(save_dir, id + "_mask.jpg"))
        save_image(denorm(face), os.path.join(save_dir, id + "_final.jpg"))
        save_image(denorm(audioOut), os.path.join(save_dir, id + "_audioOut.jpg"))
        # save_image(denorm(audio_result), os.path.join(save_dir, id + "_audio.jpg"))
        # save_image(denorm(coarse_result), os.path.join(save_dir, id + "_coarse.jpg"))
        

