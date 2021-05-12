import sys

import librosa
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import tensorflow as tf

# Global Variables
RANDOM_SEED = 1337  # Random Seed used
GENERATE_SPECTOGRAMS = True  # True if new spectograms should be generated, false if using saved ones.
TRAIN_MODEL = True  # True if a new model should be trained, false if using the saved one.
THRESHOLD = 0.25  # threshold for confidence for nocall
VALIDATION_CUTOFF = 0.8  # the percentage of examples that should be in the testing set. 1 - % for validation.

# Audio Clip Parameters
MIN_QUALITY = 4  # Minimum Xeno-Canto rating for clips used
INCLUDE_ZERO = False  # Should clips of unknown quality be used?
REQUIRED_SAMPLES = 200  # Number of good samples required to attempt labeling a bird
MAX_AUDIO_FILES = sys.maxsize  # Number of samples to include (only need to cut off for speed? we usually don't.) sys.maxsize

# Mel Spectrogram Parameters
SAMPLE_RATE = 32000  # All clips are 32,000
SIGNAL_LENGTH = 5  # number of seconds for each block of audio data
SPEC_SHAPE = (48, 128)  # height x width of saved image
FMIN = 500  # Low Pass frequency cutoff
FMAX = 12500  # High Pass frequency cutoff (500-12500 includes all sampled bird calls)


# Create 1 mel spectrogram from a 5 second audio clip
def create_spectrogram(audio_clip):
    hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
    mel_spec = librosa.feature.melspectrogram(y=audio_clip,
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

    return mel_spec


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
        mel_spec = create_spectrogram(chunk)

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


def save_list(list, filename):
    with open(filename, 'w') as filehandle:
        for listitem in list:
            filehandle.write('%s\n' % listitem)


def load_list(filename):
    # define an empty list
    list = []

    # open file and read the content in a list
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentItem = line[:-1]

            # add item to the list
            list.append(currentItem)
    return list


def generate_spectograms(train):
    # Limit the quality of samples.  Samples of quality 0 are actually of unknown quality
    query = 'rating>={}'.format(MIN_QUALITY)
    if INCLUDE_ZERO:
        query += ' | rating=0'
    train = train.query(query)

    # Eliminate birds without enough quality samples
    birds_count = {}
    for bird_species, count in zip(train.primary_label.unique(),
                                   train.groupby('primary_label')['primary_label'].count().values):
        birds_count[bird_species] = count
    most_represented_birds = [key for key, value in birds_count.items() if value >= REQUIRED_SAMPLES]

    TRAIN = train.query('primary_label in @most_represented_birds')
    LABELS = sorted(TRAIN.primary_label.unique())

    # Let's see how many species and samples we have left
    print('Number of species remaining in training data:', len(LABELS))
    print('Number of files remaining in training data:', len(TRAIN))
    print('Remaining species:', most_represented_birds)

    # Shuffle the training data and limit the number of audio files to MAX_AUDIO_FILES
    TRAIN = shuffle(TRAIN, random_state=RANDOM_SEED)[:MAX_AUDIO_FILES]

    print('Final Number of Audio Files:', len(TRAIN))

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

    # Separate out validation set, assuring no file has clips in the training and validation set
    initial_cutoff = int(len(samples) * VALIDATION_CUTOFF)
    file_at_cutoff = samples[initial_cutoff]
    num = int(file_at_cutoff[-5])
    final_cutoff = initial_cutoff - num
    new_samples = samples[:final_cutoff - 1]
    new_validate = samples[final_cutoff:]
    TRAIN_SPECS = shuffle(new_samples, random_state=RANDOM_SEED)
    VALIDATE_SPECS = shuffle(new_validate, random_state=RANDOM_SEED)

    # Save all lists
    save_list(TRAIN_SPECS, "train_specs.txt")
    save_list(VALIDATE_SPECS, "validate_specs.txt")
    save_list(LABELS, "labels.txt")

    # results
    print('SUCCESSFULLY EXTRACTED {} SPECTROGRAMS'.format(len(TRAIN_SPECS)))
    return TRAIN_SPECS, VALIDATE_SPECS, LABELS


def generate_data(SPECS, LABELS):
    specs, labels = [], []
    with tqdm(total=len(SPECS)) as pbar:
        for path in SPECS:
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
            if len(specs) == 0:
                specs = spec
            else:
                specs = np.vstack((specs, spec))

            # Add to label data
            target = np.zeros((len(LABELS)), dtype='float32')
            bird = path.split(os.sep)[-2]
            target[LABELS.index(bird)] = 1.0
            if len(labels) == 0:
                labels = target
            else:
                labels = np.vstack((labels, target))
    return specs, labels


def run_keras(train_specs, train_labels, validate_specs, validate_labels):
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

    validate_set = (validate_specs, validate_labels)
    # Let's train the model for a few epochs
    model_1.fit(train_specs,
                train_labels,
                validation_data=validate_set,
                batch_size=32,
                callbacks=callbacks,
                epochs=25)


if __name__ == "__main__":
    # Load metadata file
    train = pd.read_csv('birdclef-2021/train_metadata.csv',)

    if GENERATE_SPECTOGRAMS:
        # generate and store data
        TRAIN_SPECS, VALIDATE_SPECS, LABELS = generate_spectograms(train)
    else:
        # load previous data
        TRAIN_SPECS = load_list("train_specs.txt")
        VALIDATE_SPECS = load_list("validate_specs.txt")
        LABELS = load_list("labels.txt")

    # Parse all samples and add spectrograms into train data, primary_labels into label data
    train_specs, train_labels = generate_data(TRAIN_SPECS, LABELS)
    validate_specs, validate_labels = generate_data(VALIDATE_SPECS, LABELS)

    # Make sure your experiments are reproducible
    tf.random.set_seed(RANDOM_SEED)

    # Train Keras Model
    if TRAIN_MODEL:
        run_keras(train_specs, train_labels, validate_specs, validate_labels)

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
        mel_spec = create_spectrogram(chunk)

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
        if score > THRESHOLD:
            data['prediction'].append(species)
            scnt += 1
        else:
            data['prediction'].append('nocall')

        # Add the confidence score as well
        data['score'].append(score)

    print('SOUNDSCAPE ANALYSIS DONE. FOUND {} BIRDS.'.format(scnt))

    # Make a new data frame
    results = pd.DataFrame(data, columns=['row_id', 'prediction', 'score'])

    # Merge with ground truth so we can inspect
    gt = pd.read_csv('birdclef-2021/train_soundscape_labels.csv', )
    results = pd.merge(gt, results, on='row_id')

    # Let's look at the first 50 entries
    results.head(50)