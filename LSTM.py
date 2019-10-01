import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:2].values

# Always normalize the data as these process are hihgly computive 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

x_train = []
y_train = []
for i in range(60,1259):
    x_train.append(X[i-60:i,0])
    y_train.append(X[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


'''Using attention, did not get satifactor results using it
Got better results from LSTM, maybe works better if the sequence length is too long especially when LSTM fails
Have to check it out in a better problem
'''

INPUT_DIM = 1
TIME_STEPS = 60
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

def attention_3d_block(inputs,layer_name):
    # inputs.shape = (batch_size, time_steps, input_dim)
    name = layer_name
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    #print(a.shape)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1),name=name)(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

import os
import numpy as np
import skimage.io as io
import tensorflow.keras
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *

in1 = Input(shape=(60,1))

lstm_1 = LSTM(100,return_sequences=True, activation='tanh')(in1)
dropout_1 = Dropout(0.2)(lstm_1)


lstm_2 = LSTM(100,return_sequences=True, activation='tanh')(dropout_1)
dropout_2 = Dropout(0.2)(lstm_2)

lstm_3 = LSTM(100,return_sequences=True, activation='tanh')(lstm_2)
dropout_3 = Dropout(0.2)(lstm_3)

att = attention_3d_block(dropout_3,'att')
att_mul = Flatten()(att)

#attention = attention_3d_block(lstm_3, 'attention')

dense = Dense(1)(att_mul)

model = Model(inputs = in1, outputs = dense)

model.summary()
model = Model(inputs=model.input,
              outputs=[model.output, model.get_layer('att').output])


model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = 30, epochs=50)


ouputs = model.predict(encoded_input_text)
model_outputs = outputs[0]
attention_outputs = outputs[1]

attout = model.get_layer('att').output
attention_outputs = attout

sess = tf.Session()
with sess.as_default():
    print(type(tf.constant(attout).eval()))


class CharVal(object):
    def __init__(self, char, val):
        self.char = char
        self.val = val

    def __str__(self):
        return self.char

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
def color_charvals(s):
    r = 255-int(s.val*255)
    color = rgb_to_hex((255, r, r))
    return 'background-color: %s' % color

# if you are using batches the outputs will be in batches
# get exact attentions of chars
an_attention_output = attention_outputs[0][-60:]

# before the prediction i supposed you tokenized text
# you need to match each char and attention
char_vals = [CharVal(c, v) for c, v in zip(tokenized_text, attention_output)]
import pandas as pd
char_df = pd.DataFrame(char_vals).transpose()
# apply coloring values
char_df = char_df.style.applymap(color_charvals)
char_df



'''











