import ba

sourcedir = 'data/datasets/pascalparts/Annotations_Part/'

ppset = ba.PascalPartSet('pascpart', sourcedir, 'tail', 'aeroplane')
ppset.saveSegmentations()

traintxt = ppset.targets['parts']
valtxt = ppset.targets['parts']

