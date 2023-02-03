
from keras.layers import Input, LSTM, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import os as os
import random
import re

# The folder to load text files from.
path = 'shakes'
# Back propagation through time.
bptt = 15
# Number of most frequent tokens to use.
max_tokens = 1000
# Size of a batch used for training.
batch_size = 100
# Number of epochs used for training.
epoch_count = 10000
# Number of hidden recurrent parameters.
hidden_node_count = 100
# Tokenize by word or character?
tokenize_by_char = True
# Maximum number of batches per epoch. Useful if you 
# want to inspect the quality of prediction on a 
# set interval.
max_steps_per_epoch = 1000
# Maximum number of files to use.
max_files = 0

files_at_path = os.listdir(path)
files = []

for index in range(len(files_at_path)):
  if max_files != 0 and index == max_files:
    break
  filename = files_at_path[index]
  print("Parsing file: " + path + '/' + filename)
  contents = open(path + '/' + filename, 'r').read()
  files.append(contents)

tokenizer = Tokenizer(num_words=max_tokens, lower=False, char_level=tokenize_by_char)
tokenizer.fit_on_texts(files)
sequences = tokenizer.texts_to_sequences(files)
word_index = tokenizer.word_index

# Simplify our word index and index word to the max_tokens
word_index = { w:i for w,i in word_index.items() if i < max_tokens }
index_word = { i:w for w,i in word_index.items() }

print('Found %s unique tokens.' % len(word_index))

batch_count = 0
for sequence in sequences:
  batch_count += (len(sequence) - 1) // bptt

print("Creating " + str(batch_count) + " batches of size " + str(bptt) + " each.")

def generator(batch_size=128):
  # Our first axis iterates our batches, second axis is our 
  # time series (sequence of words in a sentence), and our
  # third axis is our one-hot encoding for each word.
  v_samples = np.zeros((batch_size, bptt, len(word_index)))
  v_targets = np.zeros((batch_size, len(word_index)))

  while 1:
    batch_index, token_index = 0, 0
    for sequence in sequences:
      for src_index in range((len(sequence) - 1) // bptt * bptt):
        v_samples[batch_index, token_index, sequence[src_index] - 1] = 1
        token_index += 1
        if token_index == bptt:
          # We capture one 'next word' for each batch.
          v_targets[batch_index, sequence[src_index + 1] - 1] = 1
          token_index = 0
          batch_index += 1
        if batch_index == batch_size:
          yield v_samples, v_targets
          batch_index = 0
          v_samples = np.zeros((batch_size, bptt, len(word_index)))
          v_targets = np.zeros((batch_size, len(word_index)))

model = Sequential()
model.add(LSTM(hidden_node_count, dropout=0.0, recurrent_dropout=0.0, input_shape=(bptt, len(word_index))))
model.add(Dense(len(word_index), activation='softmax'))
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
model.summary()

data_generator = generator(batch_size=batch_size)

for epoch in range(epoch_count):
  model.fit(data_generator, steps_per_epoch=(batch_count if max_steps_per_epoch == 0 else max_steps_per_epoch), epochs=1)

  (samples, targets) = next(data_generator)
  target_index = random.randint(0, samples.shape[0] - 1)
  input = samples[target_index:target_index + 1, :, :]
  output_string = ''

  for i in range(100):
    output = model.predict(input, verbose=0)
    output_one_hot = np.zeros_like(output)
    max_index = np.random.choice(range(len(word_index)), p=output[0].ravel())
    output_one_hot[0, max_index] = 1
    output_string = output_string + index_word[max_index + 1] + ('' if tokenize_by_char else ' ')
    input = np.concatenate((input, output_one_hot[:, np.newaxis, :]), axis=1)
    input = input[:, 1:, :]

  # Print the newly generated tokens for each epoch.
  print(output_string)
