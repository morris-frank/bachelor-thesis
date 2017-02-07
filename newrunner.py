import ba

sourcedir = 'data/datasets/pascalparts/Annotations_Part/'

ppset = ba.PascalPartSet('pascpart', sourcedir, 'stern', 'aeroplane')
# ppset.saveSegmentations()
ppset.saveBoundingBoxes('data/datasets/voc2010/JPEGImages/', negatives=2)
