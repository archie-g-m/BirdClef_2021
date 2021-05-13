import sys

import librosa
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle
from tqdm import tqdm
import tensorflow as tf
import heapq as hq

# Global Variables
RANDOM_SEED = 1337  # Random Seed used
GENERATE_SPECTOGRAMS = False  # True if new spectograms should be generated, false if using saved ones.
TRAIN_MODEL = False  # True if a new Keras model should be trained, false if using the saved one.
THRESHOLD = 0.25  # threshold for confidence for nocall
VALIDATION_CUTOFF = 0.8  # the percentage of examples that should be in the testing set. 1 - % for validation.
EVALUATE_KERAS = True  # True to evaluate the known best Keras model
SVM = False  # True to run SVM model
LOAD_DATA = False  # True to load saved data, false to generate new data.
SAVE_DATA = False  # True to save computed data

# Audio Clip Parameters
MIN_QUALITY = 4  # Minimum Xeno-Canto rating for clips used
INCLUDE_ZERO = False  # Should clips of unknown quality be used?
REQUIRED_SAMPLES = 200  # Number of good samples required to attempt labeling a bird
MAX_SIGNAL_TO_NOISE = 3  # A clips Signal to Noise ratio must be lower than this to be considered
MAX_AUDIO_FILES = sys.maxsize  # Number of samples to include (only need to cut off for speed? we usually don't.) sys.maxsize
MAX_CLIPS = 3  # The number of audio clips from each file

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


def SNR(array):
    m = array.mean()
    sd = array.std()
    return float(np.where(sd == 0, 0, m/sd))


# Define a function that splits an audio file,
# extracts spectrograms and saves them in a working directory
def get_spectrograms(filepath, primary_label, output_dir):
    # Open the file with librosa (limited to the first 15 seconds)
    sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=0)

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
    queue_list = []  # This is used to find the best signal to noise ratio chunks
    for chunk in sig_splits:
        mel_spec = create_spectrogram(chunk)

        signal_to_noise = SNR(mel_spec)

        if signal_to_noise < MAX_SIGNAL_TO_NOISE:
            # Add to priority queue
            hq.heappush(queue_list, (signal_to_noise, mel_spec))
    for i in range(0, min(MAX_CLIPS, len(queue_list)) - 1):  # first x in que
        # Save as image file
        mel_spec = hq.heappop(queue_list)[1]
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


def generate_data(SPECS, LABELS, metadata):
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

            # Add MetaData
            file = path.split(os.sep)[-1]
            file = file[:-6] + ".ogg"
            query = 'filename == "%s"' % file
            result = metadata.query(query)
            long = result.iloc[0]["longitude"]
            lat = result.iloc[0]["latitude"]
            month = int((result.iloc[0]["date"])[5:7])
            months = np.identity(12)[month-1]

            metadata_to_add = np.append(np.array([lat, long]), months)
            metadata_plus_zeroes = np.append(metadata_to_add, np.zeros(spec.shape[1] - metadata_to_add.shape[0]))
            spec = np.append(spec, np.expand_dims(metadata_plus_zeroes, axis=0), axis=0)

            # Add channel axis to 2D array
            spec = np.expand_dims(spec, -1)

            # Add new dimension for batch size
            spec = np.expand_dims(spec, 0)

            # Add to spectrogram data
            if len(specs) == 0:
                specs = spec
            else:
                specs = np.vstack((specs, spec))

            # Add to label data
            target = np.zeros((len(LABELS)), dtype='float32')
            if os.name == 'nt':
                bird = path.split(os.sep)[-2]
                bird = bird.split('/')[-1]
            elif os.name == 'posix':
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
                               input_shape=(SPEC_SHAPE[0] + 1, SPEC_SHAPE[1], 1)),
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


def evaluate_keras():

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

        site = soundscape_path[-16:-13]
        if site == "SSW":
            long = -76.45
            lat = 42.47
        elif site == "COR":
            long = -84.51
            lat = 10.12
        month = int(soundscape_path[-8:-6])
        months = np.identity(12)[month - 1]

        metadata_to_add = np.append(np.array([lat, long]), months)
        metadata_plus_zeroes = np.append(metadata_to_add, np.zeros(mel_spec.shape[1] - metadata_to_add.shape[0]))
        mel_spec = np.append(mel_spec, np.expand_dims(metadata_plus_zeroes, axis=0), axis=0)

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
    return results


def evaluate_svm(train_specs, train_labels,  validate_specs, validate_labels):
    # prepare data: one-hot training and validation labels and flatten images.
    train_specs = train_specs.reshape((train_specs.shape[0], train_specs.shape[1] * train_specs.shape[2]))
    train_specs = train_specs[:, :-(SPEC_SHAPE[1]-14)]
    validate_specs = validate_specs.reshape((validate_specs.shape[0], validate_specs.shape[1] * validate_specs.shape[2]))
    validate_specs = validate_specs[:, :-(SPEC_SHAPE[1]-14)]

    kernels = ["linear", "rbf"]
    cs = [0.01, 0.1, 1, 1e15]
    gammas = [0.01, 0.1, 1]
    best_acc = 0
    best_kernel = ""
    best_c = 0
    best_gamma = 0

    # hyperparameter tuning on the SVM
    for kernel in kernels:
        for c in cs:
            for gamma in gammas:
                thisSVC = svm.SVC(kernel=kernel, C=c, gamma=gamma)
                thisSVC.fit(train_specs, train_labels)
                predictions = thisSVC.predict(validate_specs)

                # for this to work, need to un-onehot predictions and labels into class numbers
                acc = predictions - validate_labels
                acc[acc != 0] = -1
                acc[acc == 0] = 1
                acc[acc == -1] = 0
                # Now every correct guess is a 1
                sum = acc.sum()
                # percent correct is just this sum over the size of acc
                accuracy = sum / acc.shape[0]
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_kernel, best_c, best_gamma = kernel, c, gamma

    # print best results

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
    if TRAIN_MODEL or SVM:
        if LOAD_DATA:
            train_specs = np.load("train_specs.npy")
            train_labels = np.load("train_labels.npy")
            validate_specs = np.load("validate_specs.npy")
            validate_labels = np.load("validate_labels.npy")
        else:
            # Parse all samples and add spectrograms into train data, primary_labels into label data
            train_specs, train_labels = generate_data(TRAIN_SPECS, LABELS)
            validate_specs, validate_labels = generate_data(VALIDATE_SPECS, LABELS)

        if SAVE_DATA:
            np.save("train_specs.npy", train_specs)
            np.save("train_labels.npy", train_labels)
            np.save("validate_specs.npy", validate_specs)
            np.save("validate_labels.npy", validate_labels)

    # Make sure your experiments are reproducible
    tf.random.set_seed(RANDOM_SEED)

    # Train Keras Model
    if TRAIN_MODEL:
        run_keras(train_specs, train_labels, validate_specs, validate_labels)

    if EVALUATE_KERAS:
        results = evaluate_keras()

    if SVM:
        evaluate_svm(train_specs, train_labels,  validate_specs, validate_labels)
