# Scripts that convert Keras DNN model to Java


## To start
```
python java-dnn-gen.py --model saved-models/mnist2.h5 --outputs outs
```

# Usage
```
python java-dnn-gen.py --help
usage: java-dnn-gen.py [-h] [--model MODEL [MODEL ...]] [--vgg16-model]
                       [--outputs DIR]

To convert a DNN model to JAVA program

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL [MODEL ...]
                        The input neural network model (.h5)
  --vgg16-model         vgg16 model
  --outputs DIR         the output directory

```
