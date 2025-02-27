# Audio-Driven Identity Manipulation for Face Inpainting
This is official code and dataset release for paper 'Audio-Driven Identity Manipulation for Face Inpainting' published in ACM MM 2024. [Paper](https://dl.acm.org/doi/10.1145/3664647.3680975)

Our main insight is that a person's voice carries distinct identity markers, such as age and gender, which provide an essential supplement for identity-aware face inpainting. By extracting identity information from audio as guidance, our method can naturally support tasks of identity preservation and identity swapping in face inpainting.

## Environment
```
conda create -n AudioFace python=3.8
conda activate AudioFace
pip install -r requirements.txt
```

Two pretrained models are necessary for extracting voice embedding and calculating face identity loss. Please download them from [BaiduDisk](https://pan.baidu.com/s/18vn8iVWbe3NDTV9li2rW3w) (code: 2c44) or [GoogleDrive](https://drive.google.com/drive/folders/1WAWdhpDrMkt9rHdKEOVCd4DHEc9eWvBL?usp=drive_link), and put them in main folder.

## Datasets
We post-process 3 public datasets *faceforensics*, *HDTF* and *VoxCelebID* for training and evaluation. 

Download them from [BaiduDisk](https://pan.baidu.com/s/1XNDVnHACcHFxhOu4MgLcEA) (code：ws2f) or [Google Drive](https://drive.google.com/drive/folders/1lJgW63nMiHluhO3FFFU0VztrnC61iP36?usp=drive_link), and extract them to `./Datasets/`.

We also provide some functions in `./audio_script` that help you to create your own audio dataset.

## Checkpoints
You can download our pretrained models from [BaiduDisk](https://pan.baidu.com/s/1GvraMZONbtJAS1dFA-tIXA) (code: q8xa) or [GoogleDrive](https://drive.google.com/drive/folders/1_P8nUfscgrj3717Vz5-wiDWQrRsP-ZXy?usp=drive_link), and put them in `./checkpoints/`.


## Train and Evaluation
For evaluation, please run:
```
python test.py --test_dataset faceforensics
python test.py --test_dataset HDTF
python test.py --test_dataset VoxCeleb-ID
```
We use several open-source code for calculating the identity distance, please refer to [sphereface](https://github.com/clcarwin/sphereface_pytorch), [CosFace](https://github.com/MuggleWang/CosFace_pytorch), [VGGFace](https://github.com/ZZUTK/Tensorflow-VGG-face) and [ArcFace](https://github.com/ronghuaiyang/arcface-pytorch).

For training, please run:
```
python train.py
```

We also provide 'demo.py' for fast evaluation, please download the test data from [BaiduDisk](https://pan.baidu.com/s/1_x9_kYzqbtO85uQBTEO4iQ) (code: hw5s) or [GoogleDrive](https://drive.google.com/drive/folders/1Xb_r4qjj4-NHfra_KzhEdGwD0twV_v-A?usp=drive_link). Then, please run:
```
python demo.py
```

## Acknowledgement

If you find this paper is useful, please cite:
```
@inproceedings{10.1145/3664647.3680975,
author = {Sun, Yuqi and Lin, Qing and Tan, Weimin and Yan, Bo},
title = {Audio-Driven Identity Manipulation for Face Inpainting},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3664647.3680975},
pages = {6123–6132},
numpages = {10},
keywords = {audio, face inpainting, identity, multi-modal},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```