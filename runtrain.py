import caffe
import ba.ba.caffeine.surgery as surgery

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = 'data/models/fcn8s/fcn8s-heavy-pascal.caffemodel'

# init
gpu = 1
caffe.set_device(gpu)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('data/models/fcn8s/solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
# val = np.loadtxt('../data/segvalid11.txt', dtype=str)

for _ in range(20):
    solver.step(2)
#    # score.seg_tests(solver, False, val, layer='score')
#
solver.snapshot()
