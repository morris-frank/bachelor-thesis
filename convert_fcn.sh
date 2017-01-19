#!/bin/bash

MODELPATH='data/models/fcn8s/fcn8s-heavy-pascal.caffemodel'
TRAINPROTO='data/models/fcn8s/train.prototxt'
DEPLOYPROTO='data/models/fcn8s/deploy.prototxt'
VALPROTO='data/models/fcn8s/val.tf.prototxt'

OUTPUTPATH='data/models/fcn8s/'

SCRIPTPATH='lib/caffe-tensorflow/convert.py'

python "${SCRIPTPATH}" --caffemodel="${MODELPATH}" \
                       --data-output-path="${OUTPUTPATH}" "${VALPROTO}"
