import keras.backend as K
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor  #输入变量#
        kernel_size: defualt 3, the kernel size of middle conv layer at main path #卷积核的大小#
        filters: list of integers, the filterss of 3 conv layer at main path  #卷积核的数目#
        stage: integer, current stage label, used for generating layer names #当前阶段的标签#
        block: 'a','b'..., current block label, used for generating layer names #当前块的标签#
    # Returns
        Output tensor for the block.  #返回块的输出变量#
    """
    filters1, filters2, filters3 = filters  #滤波器的名称#
    if K.image_data_format() == 'channels_last':  #代表图像通道维的位置#
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
 
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)   #卷积层，BN层，激活函数#
 
    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
 
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
 
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet8(img_width, img_height, img_channels, output_dim):
    #这里采用的权重是imagenet，可以更改，种类为1000#
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    img_input = Input(shape=(img_height, img_width, img_channels))
    #x = ZeroPadding2D((3, 3))(img_input) #对图片界面填充0，保证特征图的大小#
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(img_input) #定义卷积层#
    x = BatchNormalization(name='bn_conv1')(x) #批标准化#
    x = Activation('relu')(x) #激活函数#
    x = MaxPooling2D((3, 3), strides=(2, 2))(x) #最大池化层#
    #stage2#
    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    x = identity_block(x, 3, [64,64,256], stage=2, block='c')
    #stage3#
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    #stage4#
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    #stage5#
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
 
    #x = AveragePooling2D((7, 7), name='avg_pool')(x) #平均池化层#

    x = Flatten()(x)
    #x = Dense(classes, activation='softmax', name='fc1000')(x)
    
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Create model.
    #model = Model(inputs, x, name='resnet50')  
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model
