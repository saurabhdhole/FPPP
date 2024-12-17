

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




import tensorflow as tf
print(tf.__version__)  # Should print 2.14.0 or higher
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simple test model
model = Sequential([
    Dense(32, activation='relu', input_shape=(16,)),
    Dense(1, activation='sigmoid')
])
print(model.summary())
