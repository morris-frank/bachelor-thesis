import argparse
from ba.pascalpart import PascalPartSet
import ba.utils

DATASET = 'pascpart'
PARTPATH = 'data/datasets/pascalparts/Annotations_Part/'
IMGPATH = 'data/datasets/voc2010/JPEGImages/'


class Generator(ba.utils.NotifierClass):
    '''A class to generate training and test data from the PascalPartSet.'''

    def __init__(self, argv, **kwargs):
        '''Initialize the new experiment from the cli args

        Args:
            argv (str): The options string
        '''
        super().__init__(**kwargs)
        self.dataset_name = DATASET
        self.partset_source = PARTPATH
        self.images_source = IMGPATH
        self.negatives = 2
        self.naugment = 2
        self.parse_arguments(argv)

    def parse_arguments(self, argv):
        '''Parse the arguments for an experiment script'''
        parser = argparse.ArgumentParser(description='Runs a experiment')
        parser.add_argument('--combine', action='store_true')
        parser.add_argument('--classes', type=str, nargs='+',
                            metavar='class')
        parser.add_argument('--parts', type=str, nargs='+',
                            metavar='part')
        parser.add_argument('--default', action='store_true')
        args = parser.parse_args(args=argv)
        self.classes = args.classes
        self.parts = args.parts
        self.combine = args.combine
        self.defaulting = args.default

    def run(self):
        '''Generates the training data for that experiment'''
        ppset = PascalPartSet(
            self.dataset_name, self.partset_source, classes=self.classes,
            parts=self.parts, defaulting=self.defaulting)
        ppset.segmentations(combine=self.combine)
        ppset.bounding_boxes(self.images_source, negatives=self.negatives,
                             augment=self.naugment, combine=self.combine)
