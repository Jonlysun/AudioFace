from multiprocessing.resource_sharer import stop
import os, json
import shutil
import subprocess
from pydub import AudioSegment
import numpy as np
import glob

def change_ac_ar():
    input_path = "original_audio"
    output_path = "single_channel_16K_audio"
    os.makedirs(output_path, exist_ok=True)
    for file in sorted(os.listdir(input_path)):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.m4a', '.wav'))
        cmd = "ffmpeg -i " + input_file + " -ar 16000 " + " -ac 1 " + output_file

        subprocess.call(cmd, shell=True)

def meta_life():
    path = "conversion_dict.json"
    f = open(path)
    meta_file = json.load(f)
    return meta_file

def load_seq(name):
    path = "downloaded_videos_info"
    scene_path = os.path.join(path, name, "extracted_sequences", "0.json")
    f = open(scene_path)
    file = json.load(f)
    return file


def audio_cut():
    path = "single_channel_16K_audio"
    save_path = "single_channel_16K_audio_clip_1"    
    os.makedirs(save_path, exist_ok=True)

    metafile = meta_life()
    
    files = sorted(os.listdir(path))
    for file in files:
        file = '125.wav'
        basename = file.split('.')[0]
        name = metafile[basename][:-2]
        seq = load_seq(name)

        audio = AudioSegment.from_file(os.path.join(path, file), "wav")
        audio_time = len(audio)
        print(audio_time)
        # audio = audio[5000:-10000]      # remove the begin and end
        start_time = seq[0] // 25
        end_time = seq[-1] // 25

        print(start_time, end_time)

        index = 1
        for i in range(start_time, end_time, 6):
            begin_ = i * 1000
            end_ = begin_ + 6000
            if i + 6 > end_time:
                break
            audio_chunk = audio[begin_:end_]
            audio_chunk.export(os.path.join(save_path, basename + '_' + str(index).zfill(3) + '.wav'), format="wav")
            index += 1
        exit(0)


def audio_cut_1():
    path = "data_audio_single_channel_16K"
    save_path = "data_audio_single_channel_16K_clip_1"
    os.makedirs(save_path, exist_ok=True)
    
    files = sorted(os.listdir(path))
    for file in files[35:]:
        basename = file.split('.')[0]
        audio = AudioSegment.from_file(os.path.join(path, file), "wav")
        
        audio = audio[5000:-6000]      # remove the begin and end
        audio_time = len(audio)

        index = 0
        for i in range(0, audio_time, 6000):
            begin_ = i
            end_ = begin_ + 6000
            if i + 6000 > audio_time:
                break
            audio_chunk = audio[begin_:end_]
            audio_chunk.export(os.path.join(save_path, basename + '_' + str(index).zfill(3) + '.wav'), format="wav")
            index += 1


if __name__ == '__main__':
    audio_cut()
