import os
import warnings
import pandas as pd
import librosa
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
warnings.filterwarnings(action='ignore')

# Global vars
RANDOM_SEED = 1337
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5  # seconds
SPEC_SHAPE = (48, 128)  # height x width
FMIN = 500
FMAX = 12500
MAX_AUDIO_FILES = 1500


# Define a function that splits an audio file,
# extracts spectrograms and saves them in a working directory
def get_spectrograms(filepath, primary_label, output_dir):
    # Open the file with librosa (limited to the first 15 seconds)
    sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=0, duration=15)

    # Split signal into five second chunks
    sig_splits = []
    for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
        split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

        # End of signal?
        if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
            break

        sig_splits.append(split)

    # Extract mel spectrograms for each audio chunk
    s_cnt = 0
    saved_samples = []
    for chunk in sig_splits:

        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=chunk,
                                                  sr=SAMPLE_RATE,
                                                  n_fft=1024,
                                                  hop_length=hop_length,
                                                  n_mels=SPEC_SHAPE[0],
                                                  fmin=FMIN,
                                                  fmax=FMAX)

        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()

        # Save as image file
        save_dir = os.path.join(output_dir, primary_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] +
                                 '_' + str(s_cnt) + '.png')
        im = Image.fromarray(mel_spec * 255.0).convert("L")
        im.save(save_path)

        saved_samples.append(save_path)
        s_cnt += 1

    return saved_samples


def run_keras(model_1):
    # Build a simple model as a sequence of  convolutional blocks.
    # Each block has the sequence CONV --> RELU --> BNORM --> MAXPOOL.
    # Finally, perform global average pooling and add 2 dense layers.
    # The last layer is our classification layer and is softmax activated.
    # (Well it's a multi-label task so sigmoid might actually be a better choice)
    model_1 = tf.keras.Sequential([

        # First conv block
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(SPEC_SHAPE[0], SPEC_SHAPE[1], 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second conv block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third conv block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Fourth conv block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Global pooling instead of flatten()
        tf.keras.layers.GlobalAveragePooling2D(),

        # Dense block
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Classification layer
        tf.keras.layers.Dense(len(LABELS), activation='softmax')
    ])
    print('MODEL HAS {} PARAMETERS.'.format(model_1.count_params()))

    # Compile the model and specify optimizer, loss and metric
    model_1.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
                    metrics=['accuracy'])

    # Add callbacks to reduce the learning rate if needed, early stopping, and checkpoint saving
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      patience=2,
                                                      verbose=1,
                                                      factor=0.5),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  verbose=1,
                                                  patience=5),
                 tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                    monitor='val_loss',
                                                    verbose=0,
                                                    save_best_only=True)]

    # Let's train the model for a few epochs
    model_1.fit(train_specs,
                train_labels,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                epochs=25)

# Code adapted from:
# https://www.kaggle.com/frlemarchand/bird-song-classification-using-an-efficientnet
# Make sure to check out the entire notebook.


# Load metadata file
train = pd.read_csv('birdclef-2021/train_metadata.csv',)

# Limit the number of training samples and classes
# First, only use high quality samples
train = train.query('rating>=4')

# Second, assume that birds with the most training samples are also the most common
# A species needs at least 200 recordings with a rating above 4 to be considered common
birds_count = {}
for bird_species, count in zip(train.primary_label.unique(),
                               train.groupby('primary_label')['primary_label'].count().values):
    birds_count[bird_species] = count
most_represented_birds = [key for key, value in birds_count.items() if value >= 200]

TRAIN = train.query('primary_label in @most_represented_birds')
LABELS = sorted(TRAIN.primary_label.unique())

# Let's see how many species and samples we have left
print('NUMBER OF SPECIES IN TRAIN DATA:', len(LABELS))
print('NUMBER OF SAMPLES IN TRAIN DATA:', len(TRAIN))
print('LABELS:', most_represented_birds)

# Shuffle the training data and limit the number of audio files to MAX_AUDIO_FILES
TRAIN = shuffle(TRAIN, random_state=RANDOM_SEED)[:MAX_AUDIO_FILES]


print('FINAL NUMBER OF AUDIO FILES IN TRAINING DATA:', len(TRAIN))

# Parse audio files and extract training samples
input_dir = 'birdclef-2021/train_short_audio/'
output_dir = 'melspectrogram_dataset/'
samples = []
with tqdm(total=len(TRAIN)) as pbar:
    for idx, row in TRAIN.iterrows():
        pbar.update(1)

        if row.primary_label in most_represented_birds:
            audio_file_path = os.path.join(input_dir, row.primary_label, row.filename)
            samples += get_spectrograms(audio_file_path, row.primary_label, output_dir)

TRAIN_SPECS = shuffle(samples, random_state=RANDOM_SEED)
print('SUCCESSFULLY EXTRACTED {} SPECTROGRAMS'.format(len(TRAIN_SPECS)))

# Parse all samples and add spectrograms into train data, primary_labels into label data
train_specs, train_labels = [], []
with tqdm(total=len(TRAIN_SPECS)) as pbar:
    for path in TRAIN_SPECS:
        pbar.update(1)

        # Open image
        spec = Image.open(path)

        # Convert to numpy array
        spec = np.array(spec, dtype='float32')

        # Normalize between 0.0 and 1.0
        # and exclude samples with nan
        spec -= spec.min()
        spec /= spec.max()
        if not spec.max() == 1.0 or not spec.min() == 0.0:
            continue

        # Add channel axis to 2D array
        spec = np.expand_dims(spec, -1)

        # Add new dimension for batch size
        spec = np.expand_dims(spec, 0)

        # Add to train data
        if len(train_specs) == 0:
            train_specs = spec
        else:
            train_specs = np.vstack((train_specs, spec))

        # Add to label data
        target = np.zeros((len(LABELS)), dtype='float32')
        bird = path.split(os.sep)[-2]
        target[LABELS.index(bird)] = 1.0
        if len(train_labels) == 0:
            train_labels = target
        else:
            train_labels = np.vstack((train_labels, target))

# Make sure your experiments are reproducible
tf.random.set_seed(RANDOM_SEED)

# Train Keras Model

model = None
run_keras(model)


# Load the best checkpoint
model = tf.keras.models.load_model('best_model.h5')

# Pick a soundscape
soundscape_path = 'birdclef-2021/train_soundscapes/28933_SSW_20170408.ogg'

# Open it with librosa
sig, rate = librosa.load(soundscape_path, sr=SAMPLE_RATE)

# Store results so that we can analyze them later
data = {'row_id': [], 'prediction': [], 'score': []}

# Split signal into 5-second chunks
# Just like we did before (well, this could actually be a seperate function)
sig_splits = []
for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
    split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

    # End of signal?
    if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
        break

    sig_splits.append(split)

# Get the spectrograms and run inference on each of them
# This should be the exact same process as we used to
# generate training samples!
seconds, scnt = 0, 0
for chunk in sig_splits:
    # Keep track of the end time of each chunk
    seconds += 5

    # Get the spectrogram
    hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
    mel_spec = librosa.feature.melspectrogram(y=chunk,
                                              sr=SAMPLE_RATE,
                                              n_fft=1024,
                                              hop_length=hop_length,
                                              n_mels=SPEC_SHAPE[0],
                                              fmin=FMIN,
                                              fmax=FMAX)

    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to match the value range we used during training.
    # That's something you should always double check!
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()

    # Add channel axis to 2D array
    mel_spec = np.expand_dims(mel_spec, -1)

    # Add new dimension for batch size
    mel_spec = np.expand_dims(mel_spec, 0)

    # Predict
    p = model.predict(mel_spec)[0]

    # Get highest scoring species
    idx = p.argmax()
    species = LABELS[idx]
    score = p[idx]

    # Prepare submission entry
    data['row_id'].append(soundscape_path.split(os.sep)[-1].rsplit('_', 1)[0] +
                          '_' + str(seconds))

    # Decide if it's a "nocall" or a species by applying a threshold
    if score > 0.25:
        data['prediction'].append(species)
        scnt += 1
    else:
        data['prediction'].append('nocall')

    # Add the confidence score as well
    data['score'].append(score)

print('SOUNDSCAPE ANALYSIS DONE. FOUND {} BIRDS.'.format(scnt))

# Make a new data frame
results = pd.DataFrame(data, columns = ['row_id', 'prediction', 'score'])

# Merge with ground truth so we can inspect
gt = pd.read_csv('birdclef-2021/train_soundscape_labels.csv',)
results = pd.merge(gt, results, on='row_id')

# Let's look at the first 50 entries
results.head(50)
