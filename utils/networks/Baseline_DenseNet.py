import tensorflow as tf
import larq

class DenseNet(tf.keras.Model):

    def __init__(self, kd, arch='bdn-28', use_binary_downsampling = False, classes= 10):
        super(DenseNet, self).__init__()
        self.kd = kd
        growth_rate = 64
        if arch=='bdn-28':
            self.blocks = [6, 6, 6, 5]
            self.reduction = [2.7, 2.7, 2.2]
        elif arch=='bdn-37':
            self.blocks = [6, 8, 12, 6]
            self.reduction = [3.3, 3.3, 4]
        elif arch=='bdn-45':
            self.blocks = [6, 12, 14, 8]
            self.reduction = [2.7, 3.3, 4]
        else:
            raise ValueError(f"arch is not in the list.")

        self.classes = classes
        self.use_binary_downsampling = use_binary_downsampling
        print(f"{arch} has been loaded!")

        self.conv_input = self.conv_first()
        
        self.dense_block1 = DenseBlock(self.blocks[0], growth_rate)
        self.transition_block1 = TransitionBlock(reduction = 1/self.reduction[0], binary = self.use_binary_downsampling)
        
        self.dense_block2 = DenseBlock(self.blocks[1], growth_rate)
        self.transition_block2 = TransitionBlock(reduction = 1/self.reduction[1], binary = self.use_binary_downsampling)
        
        self.dense_block3 = DenseBlock(self.blocks[2], growth_rate)
        self.transition_block3 = TransitionBlock(reduction = 1/self.reduction[2], binary = self.use_binary_downsampling)
        
        self.dense_block4 = DenseBlock(self.blocks[3], growth_rate)
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.linear = tf.keras.layers.Dense(self.classes, kernel_initializer='he_normal',activation='softmax')

        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same',name='maxpool')

    def conv_first(self):

        return tf.keras.Sequential([
    
            tf.keras.layers.Conv2D(filters = 64, kernel_size=7, strides=2, padding='same', kernel_initializer ='he_normal', use_bias = False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),])

    def call(self, x):

        y = self.conv_input(x)
        y = self.maxpool(y)

        y = self.dense_block1(y)
        y = self.transition_block1(y)
        aux_1 = y 

        y = self.dense_block2(y)
        y = self.transition_block2(y)
        aux_2 = y

        y = self.dense_block3(y)
        y = self.transition_block3(y)
        aux_3 = y

        y = self.dense_block4(y)
        y = self.bn1(y)
        y = self.global_pool(y)
        logits = self.linear(y)
        
        if self.kd:
            return aux_1, aux_2, aux_3, logits
        
        else:
            return logits

    def model(self, input_shape):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
       


class ConvBlock(tf.keras.layers.Layer):
    
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        
        self.prelu = tf.keras.layers.PReLU()
        self.conv = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=3, padding='same', use_bias=False, kernel_initializer='glorot_normal')
        
        self.listLayers = [self.bn, self.relu, self.conv]
        # self.listLayers = [self.prelu, self.ste_sign, self.scale_beta , self.bconv, self.scale_alpha]

        
    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))
            
    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    num_classes = 0
    def __init__(self, reduction = 0.5, binary=False, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        
        TransitionBlock.num_classes += 1
        
        self.binary = binary
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.prelu = tf.keras.layers.PReLU()
        
        self.max_pool = tf.keras.layers.AvgPool2D(pool_size = 2, strides=2)
        self.reduction = reduction
    def build(self, input_shape):
        
        print(f"Binarized Downsampling Layer: {self.binary}")
        if self.binary:
            self.conv = larq.layers.QuantConv2D(filters = input_shape[-1] * self.reduction, kernel_size = 1, use_bias=False,
                                                input_quantizer='ste_sign',
                                                kernel_quantizer='ste_sign',
                                                kernel_constraint=larq.constraints.WeightClip(1.3))
        else:    
            self.conv =tf.keras.layers.Conv2D(filters = input_shape[-1] * self.reduction, kernel_size = 1, kernel_initializer = 'he_normal', use_bias=False)
            
        
    def call(self, x):
        
        x = self.batch_norm(x)
        
        x = self.max_pool(x)
           
        
        x = self.prelu(x)
        
        return self.conv(x)

