import os
import librosa
import numpy as np
import soundfile as sf


def audio_normalization(x, db=-25):
    rms = (x ** 2).mean() ** 0.5
    scalar = 10 ** (db / 20) / (rms) #scaling factor that will adjust the audio data to the desired level.
    x = x * scalar
    return x

def align_audio(clean, noise):
    if len(noise) > len(clean):
        noise = noise[:len(clean)]
    else:
        num_repeats = len(clean) // len(noise) + 1
        noise = np.tile(noise, num_repeats)[:len(clean)]

    return noise

def audioread(path, norm = True, sr=48000):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, sr = librosa.load(path, sr=sr)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(x.shape) != 1:  # multi-channel  
        x = x.T
        x = x.sum(axis=0)/x.shape[0]

    if norm:
        x = audio_normalization(x)

    return x, sr


def audiowrite(data, sr, destpath, norm=False):
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    sf.write(destpath, data, sr)
    return 'done'


def snr_mixer(clean, noise, snr):

    # Getting noise the same length as clean audio
    noise = align_audio(clean, noise)
    
    rmsclean = (clean**2).mean()**0.5
    rmsnoise = (noise**2).mean()**0.5
    
    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10 ** (snr / 20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech
