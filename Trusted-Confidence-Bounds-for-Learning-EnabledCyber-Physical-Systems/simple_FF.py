from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, MaxPooling2D, UpSampling2D, \
ZeroPadding2D, BatchNormalization, Activation, DepthwiseConv2D, GlobalAveragePooling2D

from keras.models import Sequential, Model
from keras.layers import Dense, Activation

def create_base_model(in_dims,embeddings,num_classes):
       """
       Base network to be shared.
       """
       model = Sequential()
       model.add(Dense(100, input_dim=in_dims[0]))
       model.add(Activation('relu'))
       model.add(Dense(100))
       model.add(Activation('relu'))
       model.add(Dense(100))
       model.add(Activation('relu'))
       model.add(Dense(units = embeddings,activation='relu',name='embedding'))
       model.add(Dense(units = num_classes,name='logits'))
       model.add(Activation('softmax'))
       return model
