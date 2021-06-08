import numpy as np 
import librosa 
import glob
import os 
import wave
import matplotlib.pyplot as plt
from scipy import signal
import librosa.display
from specAugment import spec_augment_tensorflow
import sys

DATAPATH = 'source_data/train/*/*.wav'
OUTPUTFOLDER = 'spectrograms/'
os.makedirs(OUTPUTFOLDER, exist_ok=True) 

NOISE = True
TIME True 
SPECAUG = False

def input_audio(wav_file, normalise):
	audio, sr = librosa.load(wav_file, sr=16000)
	if normalise:
		audio = audio * (0.7079 / np.max(np.abs(audio)))
		maxv = np.iinfo(np.int16).max
		audio = (audio * maxv).astype(np.float32)
	return audio, sr

def noise_aug(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data
            
def time_aug(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif self.shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data
    

def extract_spectrogram(filename,wav_file,OUTPUTFOLDER,NOISE,TIME,SPECAUG):
	try:
		y, sr = input_audio(wav_file,False)

	except:
		print('file corrupt')
	if NOISE:
		y = noise_aug(y, 0.01)
	if TIME: 
		y = time_aug(y, sr, 0.1, 'right')
	
	S = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=256,hop_length=128,fmax=2000)
	S = librosa.power_to_db(S)
	
	if SPECAUG: 
		shape = S.shape
		mel_spectrogram = np.reshape(S, (-1, shape[0], shape[1], 1))
		S=spec_augment_tensorflow.spec_augment(S)
	
	fig = plt.figure(num=None, figsize=(1.50, 1.50),dpi=100)
	img = librosa.display.specshow(S, sr=sr,fmax=2000,cmap='viridis')
	plt.axis('off')
	plt.margins(0)
                                                     
	plt.savefig(OUTPUTFOLDER+filename+'.png',transparent=True)
	plt.close()


if __name__ == '__main__':
	for file in glob.glob(DATAPATH):
	    filename = file.split('/')[3]
	    print(f'augment: {filename}')
	    extract_spectrogram(filename, file, OUTPUTFOLDER,NOISE,TIME,SPECAUG)

