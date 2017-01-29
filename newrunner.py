import ba.FCNPartRunner
import ba.PascalPartSet
import ba.caffeine.fcn

sourcedir = 'data/datasets/pascalparts/Annotations_Part/'

ppset = ba.PascalPartSet('pascpart', sourcedir, 'stern', 'aeroplane')
ppset.saveSegmentations()

traintxt = ppset.targets['parts']
valtxt = ppset.targets['parts']

fcn = ba.FCNPartRunner('v1', traintxt, valtxt, samples=100)
fcn.weights = 'data/models/fcn8s/fcn8s-heavy-pascal.caffemodel'
fcn.net_generator = ba.caffeine.fcn.fcn8s
fcn.baselr = 10e-4
fcn.prepare() # ?? Everythong alright then continue....
fcn.train()

fcn.solver = Null
