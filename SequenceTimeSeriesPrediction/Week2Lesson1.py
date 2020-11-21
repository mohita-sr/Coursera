import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.width=0

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()


# Sequence bias :
# Sequence bias is when the order of things can impact the selection of things.
# For example, if I were to ask you your favorite TV show, and listed
# "Game of Thrones", "Killing Eve", "Travellers" and "Doctor Who" in that order,
# you're probably more likely to select 'Game of Thrones' as you are familiar with it,
# and it's the first thing you see. Even if it is equal to the other TV shows.
# So, when training data in a dataset, we don't want the sequence to impact the training
# in a similar way, so it's good to shuffle them up.

# Separate x,y in batches of 2, shuffle buffered
# (performance - for stochastically picking records in a batch, when there is a large number of records)
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())