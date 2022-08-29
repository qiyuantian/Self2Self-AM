from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, concatenate, add, Multiply, BatchNormalization, Activation, \
                         MaxPooling3D, UpSampling3D, ELU, Dropout, LeakyReLU
from tensorflow.keras.utils import plot_model
                     
def conv3d_relu(inputs, filter_num):
    conv = Conv3D(filter_num, (3,3,3), 
                  padding='same', 
                  kernel_initializer='he_normal')(inputs)
    conv = LeakyReLU(alpha=0.1)(conv)
    
    return conv

def conv3d_dropout_relu(inputs, filter_num):
    inputs = Dropout(0.3)(inputs, training=True) # here is the difference
    conv = Conv3D(filter_num, (3,3,3), 
                  padding='same', 
                  kernel_initializer='he_normal')(inputs)
    conv = LeakyReLU(alpha=0.1)(conv)

    return conv

def unet_3d_model(num_ch = 1, output_ch = 1, filter_num=64, kinit_type='he_normal', tag='unet3d'):
    
    inputs = Input((None, None, None, num_ch)) 
    loss_weights = Input((None, None, None, 1))
    
    p0 = inputs
    
    conv1 = conv3d_relu(p0, filter_num)
    conv1 = conv3d_relu(conv1, filter_num)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = conv3d_relu(pool1, filter_num)
    conv2 = conv3d_relu(conv2, filter_num)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = conv3d_relu(pool2, filter_num)
    conv3 = conv3d_relu(conv3, filter_num)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
   
    conv4 = conv3d_relu(pool3, filter_num)
    conv4 = conv3d_relu(conv4, filter_num)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = conv3d_relu(pool4, filter_num)
    conv5 = conv3d_relu(conv5, filter_num)

    up6 = UpSampling3D(size = (2, 2, 2))(conv5)
    merge6 = concatenate([conv4,up6])
    conv6 = conv3d_dropout_relu(merge6, filter_num*2)
    conv6 = conv3d_dropout_relu(conv6, filter_num*2)
    
    up7 = UpSampling3D(size = (2, 2, 2))(conv6)
    merge7 = concatenate([conv3,up7])
    conv7 = conv3d_dropout_relu(merge7, filter_num*2)
    conv7 = conv3d_dropout_relu(conv7, filter_num*2)

    up8 = UpSampling3D(size = (2, 2, 2))(conv7)
    merge8 = concatenate([conv2,up8])
    conv8 = conv3d_dropout_relu(merge8, filter_num*2)
    conv8 = conv3d_dropout_relu(conv8, filter_num*2)

    up9 = UpSampling3D(size = (2, 2, 2))(conv8)
    merge9 = concatenate([conv1,up9])
    conv9 = conv3d_dropout_relu(merge9, filter_num*2)
    conv9 = conv3d_dropout_relu(conv9, filter_num*2)
    
    conv = concatenate([conv9,p0])
    conv = conv3d_dropout_relu(conv, filter_num)
    conv = conv3d_dropout_relu(conv, filter_num)

    conv = conv3d_dropout_relu(conv, int(filter_num / 2))
    conv = conv3d_dropout_relu(conv, int(filter_num / 2))
    
#    conv = Dropout(0.3)(conv)
    conv = Conv3D(output_ch, (1, 1, 1), padding='same',
                  activation=None, 
                  kernel_initializer='he_normal'
                  )(conv)
        
    recon = concatenate([conv, loss_weights],axis=-1)
        
    model = Model(inputs=[inputs, loss_weights], outputs=recon) 
    plot_model(model, to_file='%s.png' % tag, show_shapes=True)

    return model