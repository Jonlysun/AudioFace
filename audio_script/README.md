# AudioScript

Here we provide several founctions that you can create and manipulate audio content. 

The main files are 'audio_scipt.py' and 'utils.py'. 

In *audio_script.py*,the code defines a series of functions for processing audio files:
- Converts original audio from multi-channel to single-channel with a 16kHz sampling rate.
- Loads metadata and sequence information, cutting audio based on specific rules.
- Provides two different audio cutting methods to generate 6-second segments.

In *utils.py*, The code defines a series of utility functions and classes for the following tasks:
- Processing audio files, removing silence, and extracting log mel-spectrogram features.
- Supporting model inference for voice-to-face generation (voice2face).
- Providing batch processing capabilities to save audio features as .npy files, targeting different datasets