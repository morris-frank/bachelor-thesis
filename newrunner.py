import ba.FCNPartRunner
import ba.PascalPartSet

traintxt = 'data/datasets/pascalparts/train.txt'
valtxt = 'data/datasets/pascalparts/val.txt'
sourcedir = 'data/datasets/pascalparts/...'

ppset = PascalPartSet('pascpart', sourcedir, 'tail', 'aeroplane')
ppset.saveSegmentations()

fcn = ba.FCNPartRunner('v1', traintxt, valtxt, samples=100)
fcn.baselr = 10e-4
fcn.train()
