import ba

sourcedir = 'data/datasets/pascalparts/Annotations_Part/'

#ppset = ba.PascalPartSet('pascpart', sourcedir, 'stern', 'aeroplane')
#ppset.saveSegmentations()

traintxt = 'data/models/tmp/xaa'
valtxt = 'data/models/tmp/xab'

#traintxt = ppset.targets['parts']
#valtxt = ppset.targets['parts']

fcn = ba.FCNPartRunner('airStern_2l', traintxt, valtxt)
fcn.weights = 'data/models/fcn8s/fcn8s-heavy-pascal.caffemodel'
fcn.net_generator = ba.caffeine.fcn.fcn8s
fcn.baselr = 10e-14
fcn.epochs = 20
fcn.gpu = 2
fcn.labeldir = 'data/models/tmp/segmentations/pascpart_aeroplane_stern/'
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
    fcn.weights = 'data/models//airStern/snapshots/train_iter_{}.caffemodel'.format(lastiter)
    #fcn.solver = Null
    fcn.forwardList(list_=fcn.vallist)
    fcn.forwardList(list_=fcn.trainlist)
