from tensorflow.keras import layers, models, optimizers

def build_model(input_shape, num_actions, lr=0.00025):
    """
    Constructs and compiles a CNN Keras model to approximate the Q function.
    
    Args:
        input_shape (tuple): input shape (height, width, number of frames)
        num_actions (int): number of possible actions
    
    Returns:
        keras.Model: compiled model ready for training
    """
    # Input layer (takes as input the image made up of the last 4 frames of the Breakout game)
    inputs = layers.Input(shape=input_shape)

    # First convolutional layer (convolution filters scan the entire input image to extract specific features -> feature map)
    x = layers.Conv2D(16, kernel_size=8, strides=4, activation='relu')(inputs)

    # Pooling layer (reduces the size of the feature map by half while keeping the relevant features)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Flatten (3D feature map -> 1D vector)
    x = layers.Flatten()(x)

    # Dense layer (output layer: prediction of the Q-values associated with each possible action in the given visual state)
    outputs = layers.Dense(num_actions)(x)

    # Construction of the final model (linking input to output via the layers defined above)
    model = models.Model(inputs=inputs, outputs=outputs)

    # Loss function: mean square error, to approximate the Q-values
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse')

    return model
