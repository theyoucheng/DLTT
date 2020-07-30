
This is the neural network model for MNIST. It has validation accuracy 97.95%
on standard MNIST test data. Note that the input to this network has been normalized
into [0, 1]


Architecture

        Layer (type)                 Output Shape              Param #
        =================================================================
        conv2d_1 (Conv2D)            (None, 26, 26, 2)         20
        _________________________________________________________________
        activation_1 (Activation)    (None, 26, 26, 2)         0
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 24, 24, 4)         76
        _________________________________________________________________
        activation_2 (Activation)    (None, 24, 24, 4)         0
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 12, 12, 4)         0
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 576)               0
        _________________________________________________________________
        dense_1 (Dense)              (None, 128)               73856
        _________________________________________________________________
        activation_3 (Activation)    (None, 128)               0
        _________________________________________________________________
        dense_2 (Dense)              (None, 10)                1290
        _________________________________________________________________
        activation_4 (Activation)    (None, 10)                0
        =================================================================
        Total params: 75,242
        Trainable params: 75,242
        Non-trainable params: 0
