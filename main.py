import io
import os
import re
import shutil
import string
import time

import tensorflow as tf
import pandas as pd
import numpy as np
import logging

from keras import layers
from keras.backend import clear_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import MaxPooling1D, LSTM
from tqdm import tqdm

tqdm.pandas()


NAME = "Repo-Classification-Keras-Embeddings-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def preprocessing(readme_array):
    processed_array = []
    for ReadMe in tqdm(readme_array):
        # remove other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces).
        processed = re.sub('[^a-zA-Z ]', '', ReadMe)

        words = processed.split()

        # keep words that have length of more than 1 (e.g. gb, bb), remove those with length 1.
        processed_array.append(' '.join([word for word in words if len(word) > 1]))

    return processed_array


def preprocessingCode(readme_array):
    processed_array = []
    for Code in tqdm(readme_array):
        # remove other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces).
        processed = re.sub('[^a-zA-Z ]', '', Code)

        words = processed.split()

        # keep words that have length of more than 1 (e.g. gb, bb), remove those with length 1.
        processed_array.append(' '.join([word for word in words if len(word) > 1]))

    return processed_array


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('RepositoryDataset.csv')
df.head()
df.info()
df.describe()
df.to_numpy()
df['Code'] = df['Code'].apply(str)
df['ReadMe'] = df['ReadMe'].apply(str)
print("We have a total of {} categories".format(df['Category'].nunique()))
print(df['Category'].value_counts())
df['processedReadme'] = preprocessing(df['ReadMe'])
df['processedCode'] = preprocessingCode(df['Code'])
sentences = pd.concat([df['processedReadme'], df['processedCode']], axis=0).values
df['model_data'] = df['ReadMe'] + df['Code']



for x in range(0, df.shape[0]):
    df['model_data'].loc[x] = df['model_data'].loc[x].lower()
for x in range(0, df.shape[0]):
    df['model_data'].loc[x] = df['model_data'].loc[x].translate(str.maketrans('', '', string.punctuation))
    df['model_data'].loc[x] = df['model_data'].loc[x].replace(r'\W', "")
    df['model_data'].loc[x] = df['model_data'].loc[x].replace("\n", "")


tokenizer = Tokenizer(num_words=300, lower=True)
tokenizer.fit_on_texts(df['model_data'])
label_encoder = LabelEncoder()
label_encoder.fit(df['Category'])

def get_features(text_series):
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences,maxlen =maxlen)

maxlen = 500
max_words = 5000

y = label_encoder.fit_transform(df['Category'])
y = keras.utils.to_categorical(y)
x = get_features(df['model_data'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9000)
y_train.shape
x_train.shape
filter_length = 500
num_classes = 8
x_train = x_train.reshape(-1, 1, 353)
# print(y_train)
y_train = y_train.reshape(-1, 1, 353)
x_test = x_test.reshape(-1, 1, 89)
print(x_train.shape)
print(y_train.shape)
embedding_layer = tf.keras.layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1, 2, 3]))
result.numpy()
result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
result.shape

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

vocab_size = 10000
sequence_length = 100
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(sentences)
embedding_dim=100
model = Sequential([
    Embedding(vocab_size,embedding_dim, name="embedding"),
    Conv1D(filter_length, 3, padding='same', activation='relu', strides=1),
    MaxPooling1D(),
    LSTM(64),
    Dense(512,activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax'),
    Activation('sigmoid')
])

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(
    x_train,
    y_train,
    epochs=12,
    batch_size=8,
    callbacks=[tensorboard])
model.summary()


logging.info("Evaluating the models accuracy...")
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()