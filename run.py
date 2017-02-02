import ba

# sourcedir = 'data/datasets/pascalparts/Annotations_Part/'

traintxt = 'data/tmp/pascpart_aeroplane_stern.txt'
valtxt = 'data/tmp/pascpart_aeroplane_stern.txt'

tag = 'airStern_2lFC'

fcn = ba.FCNPartRunner(tag, traintxt, valtxt)
fcn.weights = 'data/models/fcn8s/fcn8s-heavy-pascal.caffemodel'
fcn.net_generator = ba.caffeine.fcn.fcn8s
fcn.baselr = 10e-14
fcn.epochs = 20
fcn.gpu = 1
fcn.imgdir = 'data/tmp/segmentations/pascpart_aeroplane_stern_bb/'
fcn.imgdir = 'data/tmp/segmentations/pascpart_aeroplane_stern_bb_patches/'
# fcn.imgext = 'png'
fcn.labeldir = 'data/tmp/segmentations/pascpart_aeroplane_stern_bb/'
fcn.prepare() # ?? Everythong alright then continue....
lastiter = fcn.epochs * len(fcn.trainlist)

if ba.utils.query_boolean('Wanna automatic train and go forward?'):
	fcn.train()
else:
	fcn.prepare('train')
	fcn.writeSolver()
	fcn.createSolver(fcn.target['solver'], fcn.weights, fcn.gpu)
	interp_layers = [k for k in fcn.solver.net.params.keys() if 'up' in k]
	ba.caffeine.surgery.interp(fcn.solver.net, interp_layers)

if ba.utils.query_boolean('Wanna test?'):
    fcn.weights = 'data/models/{}/snapshots/train_iter_{}.caffemodel'.format(tag, lastiter)
    #fcn.solver = Null
    fcn.forwardList(list_=fcn.vallist)
    fcn.forwardList(list_=fcn.trainlist)
