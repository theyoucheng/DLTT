
import argparse
import sys

import keras
from keras.models import *
from keras.layers import * 
from keras import *

##  DNN ==> C 
def c_convert(model):
  prog=''

  ##  input, we assume that channel is in behind
  input_shape=model.layers[0].input.shape
  ##  uint8_t 'input[channel][row][column]'
  try:
    channel=input_shape[3].value
    row=input_shape[1].value
    column=input_shape[2].value
    fc = False
  except: fc = True

  ## prog+='  ' + 'uint8_t input[{0}][{1}][{2}];\n'.format(channel, row, column)
  #prog='#include \"network.h\"\n'
  prog+='#include <unistd.h>\n'
  prog+='#include <string.h>\n'
  prog+='#include <stdio.h>\n'
  prog+='#include <stdlib.h>\n'
  if not fc:
    prog+='unsigned network(float input[{0}][{1}][{2}])\n'.format(row, column, channel)
  else:
    prog+='unsigned network(float input[{0}])\n'.format(input_shape[1].value)
  prog+='{\n'

  iw=0 ## index of weights layer

  ##  Assumption: kernel is a box shape and stride is 1
  ##  Assumption: convolutional layer is of type '2d'
  ##  Assumption: channel is put behind
  for l in range(0, len(model.layers)):

    layer=model.layers[l]
    name=layer.name
    _inp=layer.input
    _out=layer.output
    is_conv=(name.find('conv2d')>=0)
    is_dense=(name.find('dense')>=0)
    is_activation=(name.find('activation')>=0)
    is_maxpooling=(name.find('max_pooling')>=0)
    is_flatten=(name.find('flatten')>=0)
    is_dropout=False ##   we do not allow dropout
    activation=''
    if is_conv or is_dense or is_activation: activation=layer.activation
    is_relu=(str(activation).find('relu')>=0) ##  (not is_relu) ==> is_linear

    prog+='\n  //  layer {0}: {1}\n'.format(l, name)

    weights=[]
    biases=[]
    if is_dense or is_conv:
      weights=model.get_weights()[iw]
      biases=model.get_weights()[iw+1]
      iw+=2

    if is_conv:
      prog+='  float layer{3}[{0}][{1}][{2}];\n'.format(_out.shape[1].value, _out.shape[2].value, _out.shape[3].value, l)
      kernel_size=layer.kernel_size ##  Assumption: kernels are of box shape with stride 1
      for i in range(0, _out.shape[1].value):
        for j in range(0, _out.shape[2].value):
          for k in range(0, _out.shape[3].value):
            prog+='  layer{0}[{1}][{2}][{3}]={4}'.format(l, i, j, k, biases[k])
            for I in range(0, kernel_size[0]):
              for J in range(0, kernel_size[1]):
                for K in range(0, weights.shape[2]): ## number of precendent filters 
                  if l==0:
                    prog+=' + ({0})*input[{1}][{2}][{3}]'.format(weights[I][J][K][k], i+I, j+J, K)
                  else:
                    prog+=' + ({0})*layer{4}[{1}][{2}][{3}]'.format(weights[I][J][K][k], i+I, j+J, K, l-1)
            prog+=';\n'
            if is_relu:
              prog+='        if(layer{0}[{1}][{2}][{3}]<0) layer{0}[{1}][{2}][{3}]=0;\n'.format(l, i, j, k)
    elif is_flatten:
      s=_inp.shape[1].value * _inp.shape[2].value * _inp.shape[3].value
      prog+='  float layer{0}[{1}];\n'.format(l, s)
      for i in range(0, s):
        d0=int(i/(_inp.shape[2].value * _inp.shape[3].value))
        d1=int(( i % (_inp.shape[2].value * _inp.shape[3].value) ) / _inp.shape[3].value)
        d2=i - int(d0*(_inp.shape[2].value * _inp.shape[3].value)) - int(d1*_inp.shape[3].value)
        prog+='  layer{0}[{2}]=layer{1}[{3}][{4}][{5}];\n'.format(l, l-1, i, d0, d1, d2) ## there shall be a convolutional layer before the flatten layer
    elif is_dense:
      prog+='  float layer{0}[{1}];\n'.format(l, weights.shape[1])
      for i in range(0, weights.shape[1]):
        prog+='  layer{0}[{1}]={2}'.format(l, i, biases[i])
        for I in range(0, weights.shape[0]):
          if l==0:
            prog+=' + ({0})*input[{1}]'.format(weights[I][i], I)
          else:
            prog+=' + ({0})*layer{2}[{1}]'.format(weights[I][i], I, l-1)
        prog+=';\n'
        if is_relu:
          prog+='  if(layer{0}[{1}]<0) layer{0}[{1}]=0;\n'.format(l, i)

    elif is_maxpooling:
      pool_size=layer.pool_size ##  Assumption: maxpooling is of box shape with stride 1
      prog+='  float layer{3}[{0}][{1}][{2}];\n'.format(_out.shape[1].value, _out.shape[2].value, _out.shape[3].value, l)
      for i in range(0, _out.shape[1].value):
        for j in range(0, _out.shape[2].value):
          for k in range(0, _out.shape[3].value):
            prog+='  layer{0}[{1}][{2}][{3}]=0;\n'.format(l, i, j, k)
            ps0=pool_size[0]
            ps1=pool_size[1]
            for I in range(i*ps0, (i+1)*ps0):
              for J in range(j*ps1, (j+1)*ps1):
                prog+='  if(layer{0}[{5}][{6}][{4}]>layer{1}[{2}][{3}][{4}]) layer{1}[{2}][{3}][{4}]=layer{0}[{5}][{6}][{4}];\n'.format(l-1, l, i, j, k, I, J)

    elif is_activation: #is_relu:
      if is_relu:
        if len(_out.shape.as_list())>2:
          prog+='  float layer{3}[{0}][{1}][{2}];\n'.format(_out.shape[1].value, _out.shape[2].value, _out.shape[3].value, l)
          for i in range(0, _out.shape[1].value):
            for j in range(0, _out.shape[2].value):
              for k in range(0, _out.shape[3].value):
                prog+='  if(layer{0}[{2}][{3}][{4}]>0) layer{1}[{2}][{3}][{4}]=layer{0}[{2}][{3}][{4}];\n'.format(l-1, l, i, j, k)
                prog+='  else layer{0}[{1}][{2}][{3}]=0;\n'.format(l, i, j, k)
        else:
          prog+='  float layer{0}[{1}];\n'.format(l, _out.shape[1].value)
          for i in range(0, _out.shape[1].value):
            prog+='  if(layer{0}[{2}]>0) layer{1}[{2}]=layer{0}[{2}];\n'.format(l-1, l, i)
            prog+='  else layer{0}[{1}]=0;\n'.format(l, i)
      else:  ##  we only support ReLU and lienar for now
        if len(_out.shape.as_list())>2:
          prog+='  float layer{3}[{0}][{1}][{2}];\n'.format(_out.shape[1].value, _out.shape[2].value, _out.shape[3].value, l)
          for i in range(0, _out.shape[1].value):
            for j in range(0, _out.shape[2].value):
              for k in range(0, _out.shape[3].value):
                prog+='  layer{1}[{2}][{3}][{4}]=layer{0}[{2}][{3}][{4}];\n'.format(l-1, l, i, j, k)
        else:
          prog+='  float layer{0}[{1}];\n'.format(l, _out.shape[1].value)
          for i in range(0, _out.shape[1].value):
            prog+='  layer{1}[{2}]=layer{0}[{2}];\n'.format(l-1, l, i)
    else:
        if not l==len(model.layers)-1:
          print ('Unrecpgnizable DNN structure...layer {0}'.format(l))
          print (model.layers[l])
          sys.exit(0)

    if l==len(model.layers)-1: ## softmax
      prog+='  unsigned ret=0;\n'
      prog+='  float res=-100000;\n'
      for i in range(0, _out.shape[1].value):
        prog+='  if(layer{0}[{1}]>res)'.format(l, i)
        prog+='  {\n'
        prog+='    res=layer{0}[{1}];\n'.format(l,i )
        prog+='    ret={0};\n'.format(i)
        prog+='  }\n'
      prog+='  return ret;\n'
      break
  ####
  prog+='}\n'


  #index = 3
  #with open('data/mnist_train_csv.txt') as f:
  #  lines = [line.rstrip() for line in f]
  #  inp = lines[index].split(',')
  #  for i in range(0, len(inp)):
  #      inp[i] = (int)(inp[i])

  prog+='int main(int argc, char* argv[]) {\n'
  if not fc:
    prog+='  float finput[{0}][{1}][{2}];\n'.format(row, column, channel)
  else:
    prog+='  float finput[{0}];\n'.format(input_shape[1].value)
    for i in range(0, input_shape[1].value):
        prog += '  finput[{0}]={1};\n'.format(i, inp[i])
  prog+='  printf(\"%d\",network(finput));\n'

  prog+='}\n'

  #return prog
  dnn_file = open('network.c', 'w')
  #dnn_file.write(prog)
  #dnn_file.close()
  #dnn_file = open('network.h', 'w')
  #prog='#include <stdint.h>\n'
  #if not fc:
  #  prog+='unsigned network(float input[{0}][{1}][{2}]);\n'.format(row, column, channel)
  #else:
  #  prog+='unsigned network(float input[{0}]);\n'.format(input_shape[1].value)
  dnn_file.write(prog)
  dnn_file.close()


#def write_harness(isl):
#  s=0
#  if len(isl)>2: s=isl[1]*isl[2]*isl[3]
#  else: s=isl[1]
#  harness=''
#  harness+='#include "network.h"\n'
#  harness+='#include <unistd.h>\n'
#  harness+='#include <string.h>\n'
#  harness+='#include <stdio.h>\n'
#  harness+='#include <stdlib.h>\n'
#  harness+='\n'
#  harness+='int main(int argc, char* argv[]) {\n'
#  harness+='  uint8_t iinput[{0}][{1}][{2}];\n'.format(isl[1], isl[2], isl[3])
#  harness+='  float finput[{0}][{1}][{2}];\n'.format(isl[1], isl[2], isl[3])
#  harness+='  FILE *myFile;\n'
#  harness+='  if(argc == 2)\n'
#  harness+='  {\n'
#  harness+='    myFile = fopen(argv[1], "r");\n'
#  harness+='    for (int i=0; i < {0}; i++)\n'.format(isl[1])
#  harness+='      for (int j=0; j < {0}; j++)\n'.format(isl[2])
#  harness+='        for (int k=0; k < {0}; k++)\n'.format(isl[3])
#  harness+='        {\n'
#  harness+='          fscanf(myFile, "%d", &iinput[i][j][k] );\n'
#  harness+='          finput[i][j][k]=iinput[i][j][k]/255.0;\n'
#  harness+='        }\n'
#  harness+='  }\n'
#  harness+='  // else\n'
#  harness+='  // {\n'
#  harness+='  //   for (int i=0; i < {0}; i++)\n'.format(s)
#  harness+='  //   {\n'
#  harness+='  //     scanf("%d", &input[i]);\n'
#  harness+='  //   }\n'
#  harness+='  // }\n'
#  harness+='  network(finput);\n'
#  harness+='}\n'
#  h_file = open('harness.c', 'w')
#  h_file.write(harness)
#  h_file.close()
#
#  ## makefile
#  harness=''
#  harness+='CFLAGS ?= -g -w\n'
#  harness+='\n'
#  harness+='all:	harness\n'
#  harness+='\n'
#  harness+='clean:\n'
#  harness+='	rm -f harness\n'
#  harness+='\n'
#  harness+='harness: harness.c\n'
#  harness+='	${CC} ${CFLAGS} network.c harness.c -o harness\n'
#  h_file = open('Makefile', 'w')
#  h_file.write(harness)
#  h_file.close()

def main():
  parser=argparse.ArgumentParser(
          description='To convert a DNN model to the C program' )

  parser.add_argument('model', action='store', nargs='+', help='The input neural network model (.h5)')


  args=parser.parse_args()
  model = load_model(args.model[0])
  model.summary()
  c_convert(model)

  #isl=model.layers[0].input.shape.as_list() ## input shape list
  #write_harness(isl)

if __name__=="__main__":
  main()
