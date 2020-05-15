import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input,Concatenate,Lambda
from keras.layers import Conv2D, MaxPooling2D,DepthwiseConv2D
from keras.layers.merge import add
from keras import regularizers

from keras import backend as K
def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)
    


    # First residual block
    x2 = Conv2D(32,(1,1),strides=1,padding='same')(x1)
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(x2)
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Conv2D(32,(1,1),strides=1,padding='same')(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)



    x1 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(x1)
    x1 = keras.layers.normalization.BatchNormalization()(x1)
    x1 = Conv2D(32,(1,1),strides=1,padding='same')(x1)
    x1 = keras.layers.normalization.BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
  
    x3 = Concatenate(axis=-1)([x2,x1])
    x3 = Lambda(channel_shuffle)(x3)

    # Second residual block
    x4 = Conv2D(64,(1,1),strides=1,padding='same')(x3)
    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(x4)
    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Conv2D(64,(1,1),strides=1,padding='same')(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)



    x3 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(x3)
    x3 = keras.layers.normalization.BatchNormalization()(x3)
    x3 = Conv2D(64,(1,1),strides=1,padding='same')(x3)
    x3 = keras.layers.normalization.BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x5 = Concatenate(axis=-1)([x4,x3])
    x5 = Lambda(channel_shuffle)(x5)



    # Third residual block
    x6 = Conv2D(128,(1,1),strides=1,padding='same')(x5)
    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(x6)
    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Conv2D(128,(1,1),strides=1,padding='same')(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)



    x5 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(x5)
    x5 = keras.layers.normalization.BatchNormalization()(x5)
    x5 = Conv2D(128,(1,1),strides=1,padding='same')(x5)
    x5 = keras.layers.normalization.BatchNormalization()(x5)
    x5 = Activation('relu')(x5)

    x7 = Concatenate(axis=-1)([x6,x5])
    x7 = Lambda(channel_shuffle)(x7)
    #x7 = Conv2D(256,(3,3),strides=[1,1], padding='same')(x7)
    #x7 = Conv2D(256,(3,3),strides=[1,1], padding='same')(x7)
    x7 = Conv2D(256,(3,3),strides=[1,1], padding='same')(x7)
    x7 = keras.layers.normalization.BatchNormalization()(x7)
    x7 = Activation('relu')(x7)
    x9 = Conv2D(128,(3,3),strides=[1,1], padding='same')(x7)
    x9 = Flatten()(x9)
    x9 = Activation('relu')(x9)
    x9 = Dropout(0.5)(x9)

    # Steering channel
    steer = Dense(output_dim)(x9)
    #steer = Activation('tanh')(steer)
    x10 = Conv2D(128,(3,3),strides=[1,1], padding='same')(x7)
    x10 = Flatten()(x10)
    x10 = Activation('relu')(x10)
    x10 = Dropout(0.5)(x10)

    # Collision channel
    coll = Dense(output_dim)(x10)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model
