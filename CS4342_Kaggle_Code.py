import pandas as pd
import IPython.display as ipd
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('birdclef-2021/train_metadata.csv',)
train.head()

number_of_species = len(train['primary_label'].value_counts())

soundscapes = pd.read_csv('birdclef-2021/train_soundscape_labels.csv',)
soundscapes.head()

# Pick a file
audio_path = 'birdclef-2021/train_short_audio/banana/XC112602.ogg'

# Load the first 15 seconds this file using librosa
sig, rate = librosa.load(audio_path, sr=32000, offset=None, duration=15)

# The result is a 1D numpy array that conatains audio samples.
# Take a look at the shape (seconds * sample rate == 15 * 32000 == 480000)
print('SIGNAL SHAPE:', sig.shape)

plt.figure(figsize=(15, 5))
librosa.display.waveplot(sig, sr=32000)

# First, compute the spectrogram using the "short-time Fourier transform" (stft)
spec = librosa.stft(sig)

# Scale the amplitudes according to the decibel scale
spec_db = librosa.amplitude_to_db(spec, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(15, 5))
librosa.display.specshow(spec_db,
                         sr=32000,
                         x_axis='time',
                         y_axis='hz',
                         cmap=plt.get_cmap('viridis'))

###########################################################
# Mel Spectrogram

# Desired shape of the input spectrogram
SPEC_HEIGHT = 64
SPEC_WIDTH = 256

# Derive num_mels and hop_length from desired spec shape
# num_mels is easy, that's just spec_height
# hop_length is a bit more complicated
NUM_MELS = SPEC_HEIGHT
HOP_LENGTH = int(32000 * 5 / (SPEC_WIDTH - 1))  # sample rate * duration / spec width - 1 == 627

# High- and low-pass frequencies
# For many birds, these are a good choice
FMIN = 500
FMAX = 12500

# Let's get all three spectrograms
for second in [5, 10, 15]:
    # Get start and stop sample
    s_start = (second - 5) * 32000
    s_end = second * 32000

    # Compute the spectrogram and apply the mel scale
    mel_spec = librosa.feature.melspectrogram(y=sig[s_start:s_end],
                                              sr=32000,
                                              n_fft=1024,
                                              hop_length=HOP_LENGTH,
                                              n_mels=NUM_MELS,
                                              fmin=FMIN,
                                              fmax=FMAX)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Show the spec
    plt.figure(figsize=(15, 5))
    plt.title('Second: ' + str(second) + ', Shape: ' + str(mel_spec_db.shape))
    librosa.display.specshow(mel_spec_db,
                             sr=32000,
                             hop_length=HOP_LENGTH,
                             x_axis='time',
                             y_axis='mel',
                             fmin=FMIN,
                             fmax=FMAX,
                             cmap=plt.get_cmap('viridis'))
