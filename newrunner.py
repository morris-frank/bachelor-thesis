import ba.FCNPartRunner

traintxt = 'data/datasets/pascalparts/train.txt'
valtxt = 'data/datasets/pascalparts/val.txt'

fcn = ba.FCNPartRunner('v1', traintxt, valtxt, samples=100)
fcn.baselr = 10e-4
fcn.train()
