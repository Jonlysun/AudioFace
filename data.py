import os

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import glob

import shutil
import wavfile


from vad import read_wave, write_wave, frame_generator, vad_collector

def rm_sil(voice_file, vad_obj, tmp='tmp/'):
    """
       This code snippet is basically taken from the repository
           'https://github.com/wiseman/py-webrtcvad'

       It removes the silence clips in a speech recording
    """
    # audio, sample_rate = read_wave(voice_file)
    audio, sample_rate = read_wave(voice_file)
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 20, 50, vad_obj, frames)

    if os.path.exists(tmp):
       shutil.rmtree(tmp)
    os.makedirs(tmp)

    wave_data = []
    for i, segment in enumerate(segments):
        segment_file = tmp + str(i) + '.wav'
        write_wave(segment_file, segment, sample_rate)
        wave_data.append(wavfile.read(segment_file)[1])
    shutil.rmtree(tmp)

    if wave_data:
       vad_voice = np.concatenate(wave_data).astype('int16')
    return vad_voice


def get_fbank(voice, mfc_obj):
    # Extract log mel-spectrogra
    fbank = mfc_obj.sig2logspec(voice).astype('float32')

    # Mean and variance normalization of each mel-frequency 
    fbank = fbank - fbank.mean(axis=0)
    fbank = fbank / (fbank.std(axis=0)+np.finfo(np.float32).eps)

    # If the duration of a voice recording is less than 10 seconds (1000 frames),
    # repeat the recording until it is longer than 10 seconds and crop.
    full_frame_number = 1000
    init_frame_number = fbank.shape[0]
    while fbank.shape[0] < full_frame_number:
          fbank = np.append(fbank, fbank[0:init_frame_number], axis=0)
          fbank = fbank[0:full_frame_number,:]
    return fbank


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

def load_data_from_datasets(dataset_list):
    samples = []
    samples_voice = []
    samples_id = []
    for dataset in dataset_list:
        imgs_path = os.path.join('Datasets', dataset, 'Images', 'Train')
        voices_path = os.path.join('Datasets', dataset, 'Audios', 'Train')
        id_path = os.path.join('Datasets', dataset, 'IDFeat', 'Train')

        id_list = sorted(os.listdir(imgs_path))
        for id in id_list:
            img_files = sorted(glob.glob(os.path.join(imgs_path, id, '*.jpg')))
            aud_files = sorted(glob.glob(os.path.join(voices_path, id, '*.npy')))
            id_files = sorted(glob.glob(os.path.join(id_path, id, f'*.npy')))
            for img_file in img_files:
                samples.append(img_file)
                samples_voice.append(aud_files)
                samples_id.append(id_files)

    return samples, samples_voice, samples_id

class DS(data.Dataset):
    def __init__(self, transform=None, dataset_list=['faceforensics', 'HDTF', 'VoxCelebID']):


        self.samples, self.samples_voice, self.samples_id = load_data_from_datasets(dataset_list)
        
        assert len(self.samples) == len(self.samples_voice) == len(self.samples_id)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files")

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        
        sample = Image.open(sample_path).convert('RGB')

        # # load audio
        voice_paths = self.samples_voice[index]
        if len(voice_paths) == 0:
            print(sample_path)
        
        voice_id = np.random.randint(0, len(voice_paths), size=1)[0]
        voice_path = voice_paths[voice_id]
        fbank = np.squeeze(np.load(voice_path))
        fbank = torch.from_numpy(fbank.astype('float32'))

        face_id_paths = self.samples_id[index]
        if len(face_id_paths) == 0:
            print(sample_path)
        face_id_idx = np.random.randint(0, len(face_id_paths), size=1)[0]
        face_id_path = face_id_paths[face_id_idx]
        face_id = np.load(face_id_path)
        face_id = torch.from_numpy(face_id.astype('float32'))

        # ########## FFHQ ##########
        # fbank = np.zeros((64, 1000)).astype(np.float32)
        # fbank = torch.from_numpy(fbank.astype('float32'))

        if self.transform is not None:
            sample = self.transform(sample)

        mask = self.get_mask()
        # mask = self.random_mask()
        mask = torch.from_numpy(mask)

        return sample, mask, fbank, face_id
    
    @staticmethod
    def random_mask(height=256, width=256, pad=80,
                    min_stroke=5, max_stroke=10,
                    min_vertex=5, max_vertex=18,
                    min_brush_width=10, max_brush_width=30,
                    min_lenght=20, max_length=80):
        mask = np.zeros((height, width))

        max_angle = 2*np.pi
        num_stroke = np.random.randint(min_stroke, max_stroke+1)

        for _ in range(num_stroke):
            num_vertex = np.random.randint(min_vertex, max_vertex+1)
            brush_width = np.random.randint(min_brush_width, max_brush_width+1)
            start_x = np.random.randint(pad, height-pad)
            start_y = np.random.randint(pad, width-pad)

            for _ in range(num_vertex):
                angle = np.random.uniform(max_angle)
                length = np.random.randint(min_lenght, max_length+1)
                #length = np.random.randint(min_lenght, height//num_vertex)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                end_x = max(0, min(end_x, height))
                end_y = max(0, min(end_y, width))

                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1., brush_width)

                start_x, start_y = end_x, end_y

        if np.random.random() < 0.5:
            mask = np.fliplr(mask)
        if np.random.random() < 0.5:
            mask = np.flipud(mask)

        return mask.reshape((1,)+mask.shape).astype(np.float32)
    
    def get_mask(self):

        mask = []
        IMAGE_SIZE = 256

        up = IMAGE_SIZE // 4 + 25
        down = IMAGE_SIZE - IMAGE_SIZE // 4 + 25
        left = IMAGE_SIZE // 4
        right = IMAGE_SIZE - IMAGE_SIZE // 4
            
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        m[up:down, left:right] = 1
        mask = np.array(m)
        
        return mask.reshape((1,)+mask.shape).astype(np.float32) #, np.array(points)
