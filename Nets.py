import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.layers import Layer
from keras.layers import Input, Dense, GRU, Attention, Concatenate, Dropout, BatchNormalization, LayerNormalization, Conv1D, Flatten, Reshape, Activation, Embedding, GlobalMaxPooling1D, SeparableConv1D, MaxPooling1D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


import numpy as np

from Metrics import specificity, f1_score, weighted_f1_score, specificity_multilabel, weighted_f1_score_multilabeled

class CustomConvLayer(Layer):
    def __init__(self, filter_num, filter_size, **kwargs):
        super(CustomConvLayer, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.conv1 = Conv1D(filters=filter_num, kernel_size=filter_size, use_bias=True, activation='relu', name='conv1')
        self.conv2 = Conv1D(filters=filter_num, kernel_size=filter_size, use_bias=False, strides=2, name='conv2')
        self.bn = BatchNormalization(name='bn')
        self.activation = Activation(activation='relu', name='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConvLayer, self).get_config()
        config.update({
            'filter_num': self.filter_num,
            'filter_size': self.filter_size
        })
        return config
    
    def set_weights(self, weight_list):
        self.conv1.set_weights(weight_list[:2])
        self.conv2.set_weights(weight_list[2:3])
        self.bn.set_weights(weight_list[3:])

    def freeze_layers(self):
        self.conv1.trainable = False
        self.conv2.trainable = False
        self.bn.trainable    = False

    def unfreeze_layers(self):
        self.conv1.trainable = True
        self.conv2.trainable = True
        self.bn.trainable    = True

def make_single_conv_model(block_num, max_len, vocab_size, filter_num=64, filter_size=3, unit_num=32):
    # Initialize all layers
    input_tensor = Input(shape=(1, max_len, 1), name='input_tensor')
    
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding')

    block_list = []
    for i in range(block_num):
        depth_factor = i if i < 4 else 3
        block_list.append(CustomConvLayer((2**depth_factor) * filter_num, filter_size))
    
    flatten_layer  = Flatten()
    
    fc_layer_1 = Dense(units=unit_num, activation='relu', name='fc_layer_1')
    bn_layer   = BatchNormalization(name='bn_fc_layer')
    fc_layer_2 = Dense(units=unit_num, activation='relu', name='fc_layer_2')
    
    output_layer = Dense(units=1, activation='sigmoid', name='output_layer')
    
    # Apply all layers
    x = embedding_layer(input_tensor[:, :, :, 0])
    for a_block in block_list:
        x = a_block(x)
    
    x = flatten_layer(x)

    x = fc_layer_1(x)
    x = bn_layer(x)
    x = fc_layer_2(x)
    
    output_tensor = output_layer(x)

    # Create the primary prediction model
    primary_model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Compile the primary model
    primary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), specificity, tf.keras.metrics.Precision(name='precision'), weighted_f1_score])
    
    return primary_model

def make_triplet_conv_model(block_num, max_len, vocab_size, filter_num=64, filter_size=3, unit_num=32):
    # Initialize all layers
    input_tensor = Input(shape=(1, max_len, 3), name='input_tensor')

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding')

    block_list = []
    for i in range(block_num):
        depth_factor = i if i < 4 else 3
        block_list.append(CustomConvLayer((2**depth_factor) * filter_num, filter_size))
    
    flatten_layer  = Flatten()

    attention_layer = Attention(name='class_attention')
    concatenation_layer = Concatenate(name='concatenated_features')
    
    fc_layer_1   = Dense(units=unit_num, activation='relu', name='fc_layer_1')
    bn_layer_1   = BatchNormalization(name='bn_fc_layer_1')
    fc_layer_2   = Dense(units=unit_num, activation='relu', name='fc_layer_2')
    bn_layer_2   = BatchNormalization(name='bn_fc_layer_2')
    output_layer = Dense(units=1, activation='sigmoid', name='output_layer')

    # Apply all layers
    processed_channels = []
    cam_output = []
    for i in range(3):
        x = embedding_layer(input_tensor[:, :, :, i])
        for a_block in block_list:
            x = a_block(x)
        cam_output.append(x)
        
        x = flatten_layer(x)
        processed_channels.append(x)

    x = attention_layer(processed_channels)
    x = concatenation_layer([x] + processed_channels)

    x = fc_layer_1(x)
    x = bn_layer_1(x)
    x = fc_layer_2(x)
    x = bn_layer_2(x)
    
    output_tensor = output_layer(x)

    # Create the primary prediction model
    conv_model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Compile the primary model
    conv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), specificity])

    # Create the CAM model
    cam_model = Model(inputs=input_tensor, outputs=cam_output)
        
    # Create the classifier model
    class_model_inputs    = [Input(shape=output_shape[1:], name=f'class_model_input_{i}') for i, output_shape in enumerate(cam_model.output_shape)]
    flattened_inputs      = [flatten_layer(z) for z in class_model_inputs]
    attention_output      = attention_layer(flattened_inputs)
    concatenated_features = concatenation_layer([attention_output] + flattened_inputs)
    fc_output_1           = fc_layer_1(concatenated_features)
    bn_output_1           = bn_layer_1(fc_output_1)
    fc_output_2           = fc_layer_2(bn_output_1)
    bn_output_2           = bn_layer_2(fc_output_2)
    class_model_output    = output_layer(fc_output_2)
    
    # Create the class model
    class_model = Model(inputs=class_model_inputs, outputs=class_model_output)

    return conv_model, cam_model, class_model   

def make_single_conv_model_from_model(a_model, max_len, filter_num=64, filter_size=3, unit_num=32, vocab_size=8):
    # Define one input layer for the three channels
    input_tensor = Input(shape=(1, max_len, 1), name='input_tensor')

    # Embedding layer for all channels
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding', weights=a_model.layers[2].get_weights())

    # Create instances of the custom layer
    custom_layer_1 = CustomConvLayer(filter_num, filter_size)
    custom_layer_2 = CustomConvLayer(2*filter_num, filter_size)
    custom_layer_3 = CustomConvLayer(4*filter_num, filter_size)
    custom_layer_4 = CustomConvLayer(8*filter_num, filter_size)
    #custom_layer_5 = CustomConvLayer(8*filter_num, filter_size)
    # custom_layer_6 = CustomConvLayer(8*filter_num, filter_size)
    #custom_layer_7 = CustomConvLayer(8*filter_num, filter_size)

    flatten_layer  = Flatten()

    # Process each channel through the embedding and custom layer
    #cam_output = []
    
    channel_input = embedding_layer(input_tensor[:, :, :, 0])
    processed_channel_1 = custom_layer_1(channel_input)
    processed_channel_2 = custom_layer_2(processed_channel_1)
    processed_channel_3 = custom_layer_3(processed_channel_2)
    processed_channel_4 = custom_layer_4(processed_channel_3)
    #processed_channel_5 = custom_layer_5(processed_channel_4)
    # processed_channel_6 = custom_layer_6(processed_channel_5)

    # embedded_seq = bn_embed_layer(embed_layer(embed_attention_layer(flatten_layer(processed_channel_6))))
    embedded_seq = flatten_layer(processed_channel_4)
    #embedded_seq = GlobalMaxPooling1D()(tf.squeeze(processed_channel_4, axis=1))

    #cam_output.append(embedded_seq)
        
        # Original
        # cam_output.append(processed_channel_6)
        # flatten_output = flatten_layer(processed_channel_6)
        # processed_channels.append(flatten_output)
        
        # processed_channels.append(encode_layer(flatten_output))
        
    # # Attention mechanism
    #attention_layer = Attention(name='attention')([embedded_seq, embedded_seq])
    #attention_output = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=1)(embedded_seq, embedded_seq)
    #concatenated_features = Concatenate(name='concatenated_features')([attention_layer] + [embedded_seq])

    # Fully connected layers
    x   = Dense(units=unit_num, name='fc_layer_1')(embedded_seq)
    x   = BatchNormalization(name='bn_fc_layer_1')(x)
    x   = Activation(activation='relu', name='relu_1')(x)
    
    x   = Dense(units=unit_num, name='fc_layer_2')(x)
    x   = BatchNormalization(name='bn_fc_layer_2')(x)
    x   = Activation(activation='relu', name='relu_2')(x)

    output_layer = Dense(units=1, activation='sigmoid', name='output_layer')(x)

    # Create the primary prediction model
    primary_model = Model(inputs=input_tensor, outputs=output_layer)
    
    # Compile the primary model
    primary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), specificity, tf.keras.metrics.Precision(name='precision'), weighted_f1_score])

    custom_layer_1.set_weights(a_model.layers[3].get_weights())
    custom_layer_2.set_weights(a_model.layers[4].get_weights())
    custom_layer_3.set_weights(a_model.layers[5].get_weights())
    custom_layer_4.set_weights(a_model.layers[6].get_weights())

    return primary_model
    
def ensemble_predict(model_list, scan_tensor):
    pred_list = []
    for a_model in model_list:
        pred_list.append(a_model.predict(scan_tensor))
    return np.concatenate(pred_list, axis=1)


def make_single_recurrent_model(max_len, vocab_size=8, dense_unit_num=32, gru_unit_num=2, activation='tanh'):
    # Initialize all layer
    input_tensor = Input(shape=(1, max_len, 1), name='input_tensor')
    
    # Embedding layer for all channels
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding')
    
    # Reshape layer to turn a 4 dim to 3 dim
    reshape_layer = Reshape((max_len, 1))
    
    bn1 = BatchNormalization(name='bn1')
    bn2 = BatchNormalization(name='bn2')
    activation1 = Activation(activation=activation, name=f'{activation}_1')
    activation2 = Activation(activation=activation, name=f'{activation}_2')
    
    # 2 recurrent layers
    gru_layer_1 = GRU(units=gru_unit_num, return_sequences=True, unroll=True,  name='gru_layer_1', use_bias=False) #  recurrent_dropout=0.2,
    gru_layer_2 = GRU(units=gru_unit_num, unroll=True, name='gru_layer_2', use_bias=False) # recurrent_dropout=0.2, 
    
    fc_layer_1 = Dense(units=dense_unit_num, activation='relu', name='fc_layer_1')
    bn_layer_1 = BatchNormalization(name='bn_fc_layer')
    fc_layer_2 = Dense(units=dense_unit_num, activation='relu', name='fc_layer_2')
    bn_layer_2 = BatchNormalization(name='bn_fc_layer_2')
    output_layer = Dense(units=1, activation='sigmoid', name='output_layer')
    
    x = embedding_layer(input_tensor)
    x = reshape_layer(x)
    
    x = gru_layer_1(x)
    x = bn1(x)
    x = activation1(x)
    
    x = gru_layer_2(x)
    x = bn2(x)
    x = activation2(x)
    
    x = fc_layer_1(x)
    x = bn_layer_1(x)
    x = fc_layer_2(x)
    x = bn_layer_2(x)
    
    output_tensor = output_layer(x)
    
    # Create the primary prediction model
    primary_model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Compile the primary model
    primary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), specificity, tf.keras.metrics.Precision(name='precision'), weighted_f1_score])

    return primary_model


def make_triplet_vanilla_conv_model(block_num, max_len, vocab_size, filter_num=64, filter_size=3, unit_num=32):
    # Initialize all layers
    input_tensor = Input(shape=(1, max_len, 3), name='input_tensor')

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding')

    block_list = []
    for i in range(block_num):
        depth_factor = i if i < 4 else 3
        block_list.append(CustomConvLayer((2**depth_factor) * filter_num, filter_size))
    
    flatten_layer  = Flatten()

    attention_layer = Attention(name='class_attention')
    concatenation_layer = Concatenate(name='concatenated_features')
    
    fc_layer_1   = Dense(units=unit_num, activation='relu', name='fc_layer_1')
    bn_layer_1   = BatchNormalization(name='bn_fc_layer_1')
    fc_layer_2   = Dense(units=unit_num, activation='relu', name='fc_layer_2')
    bn_layer_2   = BatchNormalization(name='bn_fc_layer_2')
    output_layer = Dense(units=1, activation='sigmoid', name='output_layer')

    # Apply all layers
    processed_channels = []
    cam_output = []
    embedding_output = []
    for i in range(3):
        x = embedding_layer(input_tensor[:, :, :, i])
        embedding_output.append(x)
        for a_block in block_list:
            x = a_block(x)
        cam_output.append(x)
        
        x = flatten_layer(x)
        processed_channels.append(x)

    x = attention_layer(processed_channels)
    x = concatenation_layer([x] + processed_channels)

    x = fc_layer_1(x)
    x = bn_layer_1(x)
    x = fc_layer_2(x)
    x = bn_layer_2(x)
    
    output_tensor = output_layer(x)

    # Create the primary prediction model
    conv_model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Compile the primary model
    conv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), specificity])

    # Create the CAM model
    cam_model = Model(inputs=input_tensor, outputs=cam_output)
    
    # Creat two models for the Vanilla Gradient
    vanilla_base_model  = Model(inputs=input_tensor, outputs=embedding_output)
    vanilla_class_input = [Input(shape=output_shape[1:], name=f'vanilla_class_input_{i}') for i, output_shape in enumerate(vanilla_base_model.output_shape)]
    vanilla_list = []
    for z in vanilla_class_input:
        for a_block in block_list:
            z = a_block(z)
        z = flatten_layer(z)
        vanilla_list.append(z)
    
    z = attention_layer(vanilla_list)
    z = concatenation_layer([z] + vanilla_list)

    z = fc_layer_1(z)
    z = bn_layer_1(z)
    z = fc_layer_2(z)
    z = bn_layer_2(z)
    
    vanilla_output_tensor = output_layer(z)    
    vanilla_class_model   = Model(inputs=vanilla_class_input, outputs=vanilla_output_tensor)
        
    # Create the classifier model
    class_model_inputs    = [Input(shape=output_shape[1:], name=f'class_model_input_{i}') for i, output_shape in enumerate(cam_model.output_shape)]
    flattened_inputs      = [flatten_layer(z) for z in class_model_inputs]
    attention_output      = attention_layer(flattened_inputs)
    concatenated_features = concatenation_layer([attention_output] + flattened_inputs)
    fc_output_1           = fc_layer_1(concatenated_features)
    bn_output_1           = bn_layer_1(fc_output_1)
    fc_output_2           = fc_layer_2(bn_output_1)
    bn_output_2           = bn_layer_2(fc_output_2)
    class_model_output    = output_layer(fc_output_2)
    
    # Create the class model
    class_model = Model(inputs=class_model_inputs, outputs=class_model_output)

    return conv_model, cam_model, class_model, vanilla_base_model, vanilla_class_model

#Train a model with an output layer of two neurons instead of one
#Uses convolutional weights from an already trained model
#Weigts of the dense layer are intialized randomly
def make_triplet_vanilla_conv_model_two_class(a_model, block_num, max_len, vocab_size, filter_num=64, filter_size=3, unit_num=32):
    # Initialize all layers
    input_tensor = Input(shape=(1, max_len, 3), name='input_tensor')
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding', weights=a_model.layers[4].get_weights() )

    block_list = []
    for i in range(block_num):
        depth_factor = i if i < 4 else 3
        block_list.append(CustomConvLayer((2**depth_factor) * filter_num, filter_size))
    
    flatten_layer  = Flatten()

    attention_layer = Attention(name='class_attention')
    concatenation_layer = Concatenate(name='concatenated_features')
    
    fc_layer_1   = Dense(units=unit_num, activation='relu', name='fc_layer_1')
    bn_layer_1   = BatchNormalization(name='bn_fc_layer_1')
    fc_layer_2   = Dense(units=unit_num, activation='relu', name='fc_layer_2')
    bn_layer_2   = BatchNormalization(name='bn_fc_layer_2')
    output_layer = Dense(units=2, activation='softmax', name='output_layer')

    # Apply all layers
    processed_channels = []
    cam_output = []
    embedding_output = []
    for i in range(3):
        x = embedding_layer(input_tensor[:, :, :, i])
        embedding_output.append(x)
        for a_block in block_list:
            x = a_block(x)
        cam_output.append(x)
        
        x = flatten_layer(x)
        processed_channels.append(x)

    x = attention_layer(processed_channels)
    x = concatenation_layer([x] + processed_channels)

    x = fc_layer_1(x)
    x = bn_layer_1(x)
    x = fc_layer_2(x)
    x = bn_layer_2(x)
    
    output_tensor = output_layer(x)

    # Create the primary prediction model
    conv_model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Compile the primary model
    conv_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), specificity])

    # Create the CAM model
    cam_model = Model(inputs=input_tensor, outputs=cam_output)
    
    # Creat two models for the Vanilla Gradient
    vanilla_base_model  = Model(inputs=input_tensor, outputs=embedding_output)
    vanilla_class_input = [Input(shape=output_shape[1:], name=f'vanilla_class_input_{i}') for i, output_shape in enumerate(vanilla_base_model.output_shape)]
    vanilla_list = []
    for z in vanilla_class_input:
        for a_block in block_list:
            z = a_block(z)
        z = flatten_layer(z)
        vanilla_list.append(z)
    
    z = attention_layer(vanilla_list)
    z = concatenation_layer([z] + vanilla_list)

    z = fc_layer_1(z)
    z = bn_layer_1(z)
    z = fc_layer_2(z)
    z = bn_layer_2(z)
    
    vanilla_output_tensor = output_layer(z)    
    vanilla_class_model   = Model(inputs=vanilla_class_input, outputs=vanilla_output_tensor)
        
    # Create the classifier model
    class_model_inputs    = [Input(shape=output_shape[1:], name=f'class_model_input_{i}') for i, output_shape in enumerate(cam_model.output_shape)]
    flattened_inputs      = [flatten_layer(z) for z in class_model_inputs]
    attention_output      = attention_layer(flattened_inputs)
    concatenated_features = concatenation_layer([attention_output] + flattened_inputs)
    fc_output_1           = fc_layer_1(concatenated_features)
    bn_output_1           = bn_layer_1(fc_output_1)
    fc_output_2           = fc_layer_2(bn_output_1)
    bn_output_2           = bn_layer_2(fc_output_2)
    class_model_output    = output_layer(fc_output_2)
    
    # Create the class model
    class_model = Model(inputs=class_model_inputs, outputs=class_model_output)

    block_list[0].set_weights(a_model.layers[5].get_weights())
    block_list[1].set_weights(a_model.layers[6].get_weights())
    block_list[2].set_weights(a_model.layers[7].get_weights())
    block_list[3].set_weights(a_model.layers[8].get_weights())
    
    return conv_model, cam_model, class_model, vanilla_base_model, vanilla_class_model

def make_pair_conv_model(block_num, max_len, vocab_size, filter_num=64, filter_size=3, unit_num=32):
    # Initialize all layers
    input_tensor = Input(shape=(1, max_len, 2), name='input_tensor')

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding')

    block_list = []
    for i in range(block_num):
        depth_factor = i if i < 4 else 3
        block_list.append(CustomConvLayer((2**depth_factor) * filter_num, filter_size))
    
    flatten_layer  = Flatten()

    attention_layer = Attention(name='class_attention')
    concatenation_layer = Concatenate(name='concatenated_features')
    
    fc_layer_1   = Dense(units=unit_num, activation='relu', name='fc_layer_1')
    bn_layer_1   = BatchNormalization(name='bn_fc_layer_1')
    fc_layer_2   = Dense(units=unit_num, activation='relu', name='fc_layer_2')
    bn_layer_2   = BatchNormalization(name='bn_fc_layer_2')
    output_layer = Dense(units=1, activation='sigmoid', name='output_layer')

    # Apply all layers
    processed_channels = []
    cam_output = []
    for i in range(2):
        x = embedding_layer(input_tensor[:, :, :, i])
        for a_block in block_list:
            x = a_block(x)
        cam_output.append(x)
        
        x = flatten_layer(x)
        processed_channels.append(x)

    x = attention_layer([processed_channels[0], processed_channels[1], processed_channels[1]])
    x = concatenation_layer([x] + processed_channels)

    x = fc_layer_1(x)
    x = bn_layer_1(x)
    x = fc_layer_2(x)
    x = bn_layer_2(x)
    
    output_tensor = output_layer(x)

    # Create the primary prediction model
    conv_model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Compile the primary model
    conv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), specificity])

    # Create the CAM model
    cam_model = Model(inputs=input_tensor, outputs=cam_output)
        
    # Create the classifier model
    class_model_inputs    = [Input(shape=output_shape[1:], name=f'class_model_input_{i}') for i, output_shape in enumerate(cam_model.output_shape)]
    flattened_inputs      = [flatten_layer(z) for z in class_model_inputs]
    attention_output      = attention_layer(flattened_inputs)
    concatenated_features = concatenation_layer([attention_output] + flattened_inputs)
    fc_output_1           = fc_layer_1(concatenated_features)
    bn_output_1           = bn_layer_1(fc_output_1)
    fc_output_2           = fc_layer_2(bn_output_1)
    bn_output_2           = bn_layer_2(fc_output_2)
    class_model_output    = output_layer(fc_output_2)
    
    # Create the class model
    class_model = Model(inputs=class_model_inputs, outputs=class_model_output)

    return conv_model, cam_model, class_model

def make_multilabel_single_conv_model(block_num, label_num, max_len, vocab_size, filter_num=64, filter_size=3, unit_num=32):
    # Initialize all layers
    input_tensor = Input(shape=(1, max_len, 1), name='input_tensor')
    
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding')

    block_list = []
    for i in range(block_num):
        depth_factor = i if i < 4 else 3
        block_list.append(CustomConvLayer((2**depth_factor) * filter_num, filter_size))
    
    flatten_layer  = Flatten()
    
    fc_layer_1 = Dense(units=label_num, activation='relu', name='fc_layer_1') #unit_num
    bn_layer   = BatchNormalization(name='bn_fc_layer')
    fc_layer_2 = Dense(units=unit_num, activation='relu', name='fc_layer_2')
    
    output_layer = Dense(units=label_num, activation='sigmoid', name='output_layer')
    
    # Apply all layers
    x = embedding_layer(input_tensor[:, :, :, 0])
    for a_block in block_list:
        x = a_block(x)
    
    x = flatten_layer(x)

    x = fc_layer_1(x)
    x = bn_layer(x)
    x = fc_layer_2(x)
    
    output_tensor = output_layer(x)

    # Create the primary prediction model
    primary_model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Compile the primary model
    primary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Recall(name='recall'),specificity_multilabel, tf.keras.metrics.Precision(name='precision'), weighted_f1_score_multilabeled])
    
    return primary_model

def make_old_conv_model(max_len, filter_num=64, filter_size=3, unit_num=32, vocab_size=None):
    '''
    A classifier that processes the three sequences together
    '''    

    embedding_layer=Embedding(input_dim=vocab_size, output_dim=1, mask_zero=True, name='embedding')
    
    inputs = Input(shape=(1, max_len, 3), name='input_tensor')
    print(inputs.shape)

    l = []
    for i in range(3):
        embedded_seq = embedding_layer(inputs[:, :, :, i])
        l.append(embedded_seq)    
    z = Concatenate(axis=-1)(l)
    
    z = tf.squeeze(inputs, axis=1)
    print(z.shape)

    z = SeparableConv1D(filters=filter_num, kernel_size=filter_size, use_bias=False, name='conv1')(z)
    z = BatchNormalization()(z)
    z = Activation(activation='selu')(z)
    z = MaxPooling1D(pool_size=2)(z)

    z = SeparableConv1D(filters=filter_num*2, kernel_size=filter_size, use_bias=False, name='conv2')(z)
    z = BatchNormalization()(z)
    z = Activation(activation='selu')(z)
    z = MaxPooling1D(pool_size=2)(z)
    
    z = SeparableConv1D(filters=filter_num*4, kernel_size=filter_size, use_bias=False, name='conv3')(z)
    z = BatchNormalization()(z)
    z = Activation(activation='selu')(z)
    z = MaxPooling1D(pool_size=2)(z)
    
    #     z = SeparableConv1D(filters=filter_num*8, kernel_size=filter_size, use_bias=False, name='conv4')(z)
    #     z = BatchNormalization()(z)
    #     z = Activation(activation='selu')(z)
    #     z = MaxPooling1D(pool_size=2)(z)

    #     z = SeparableConv1D(filters=filter_num*16, kernel_size=filter_size, use_bias=False, name='conv5')(z)
    #     z = BatchNormalization()(z)
    #     z = Activation(activation='selu')(z)
    #     z = GlobalMaxPooling1D()(z)
        
    z = SeparableConv1D(filters=filter_num*8, kernel_size=filter_size, use_bias=False, name='conv5')(z)
    z = BatchNormalization()(z)
    z = Activation(activation='selu')(z)
    z = GlobalMaxPooling1D()(z)
    
    #z = keras.layers.Flatten()(z)
    
    z = Dense(unit_num, use_bias=False, name='dense1')(z)
    z = BatchNormalization()(z)
    z = Activation(activation='selu')(z)
    
    outputs = Dense(1, activation='sigmoid', name='dense2')(z)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), specificity])
    
    return model