import os.path
from src.PascalPart import PascalPart
from tqdm import tqdm
import warnings

class SetList(object):
    """docstring for SetList."""
    def __init__(self, source=''):
        super(SetList, self).__init__()
        self.source = source
        if source != '':
            self.load()

    def load(self):
        open(self.source, 'a').close()
        with open(self.source) as f:
            self.list = f.read().splitlines()

    def save(self):
        with open(self.source, 'w') as f:
            for row in self.list:
                f.write("{}\n".format(row))
                print('List {} written...'.format(self.source))

    def addPreSuffix(self, prefix, suffix):
        self.list = [prefix + x + suffix for x in self.list]

    def rmPreSuffix(self, prefix, suffix):
        self.list = [x[len(prefix):-len(suffix)] for x in self.list]

    def genPartList(self, classname, parts):
        print('generate Parts list for {} in {}'.format(parts, classname))
        bn, en = os.path.splitext(self.source)
        plist = SetList('_'.join([bn, classname]) + '_' + '_'.join(parts) + en)
        for i in tqdm(range(len(self.list) - 1)):
            pp = PascalPart(self.list[i])
            if classname and pp.classname != classname:
                continue

            if not any(x in pp.parts for x in parts):
                continue

            plist.content.append(self.list[i])

        return plist

    def each(self, callback):
        if not callable(callback):
            warnings.warn('Not callable object')
            return
        print('Calling each object in SetList....')
        for i in tqdm(range(len(self.list) - 1)):
            callback(self.list[i])
