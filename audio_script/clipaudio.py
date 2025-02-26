import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np
 
source_dir = "./original_audio/"
audio_list = os.listdir(source_dir)
audio_list.sort()

target_dir = "./melspectrogram_tmp/"   

if not os.path.exists(target_dir): 
    os.makedirs(target_dir)

for file in audio_list:
    
    print(file)

    sig, fs = librosa.load(source_dir + file)
    # make pictures name 
    save_path = target_dir + file[:-4] + ".png"
    
    # pylab.axis('off') # no axis
    # pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    print(S.shape)

    # librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    # pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    # pylab.close()

    