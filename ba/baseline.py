import skimage.feature
import sklearn.svm
import ba.set
import ba.eval
import ba.utils
import random
from tqdm import tqdm
import scipy.misc
import numpy as np
import time


class Baseline(object):
    def __init__(self, tag):
        self.tag = tag
        self.model = sklearn.svm.LinearSVC()
        self.ks = 224
        self.patch_size = (self.ks, self.ks)
        self.images = 'data/datasets/voc2010/JPEGImages/'
        self.negatives = 'data/tmp/var_neg/'
        self.slicefile = 'data/tmp/pascpart/patches/' + self.tag + '/seg.yaml'
        self.ext = 'jpg'
        self.ppI = 10
        self.set = ba.set.SetList('data/tmp/pascpart/' + self.tag + '.txt')
        self.out = 'data/tmp/baseline/' + self.tag
        self.mean = 0
        # self.set.add_pre_suffix(self.images, '.jpg')
        # self.mean = self.set.calculate_mean()
        # self.set.rm_pre_suffix(self.images, '.jpg')

    def imread(self, path):
        im = scipy.misc.imread(path)
        im = np.array(im, dtype=np.float32)
        im = im[:, :, ::-1]
        return im

    def features(self, image, feature_vector=True):
        # Images are in BGR order !!
        greyscale = 0.2125 * image[2] + 0.7154 * image[1] + 0.0721 * image[0]
        hog_array = skimage.feature.hog(greyscale, block_norm='L2-Hys',
                                        feature_vector=feature_vector)
        return hog_array

    def pickImages(self, nsamples):
        return random.sample(self.set.list, nsamples)

    def test_single_scale(self, im, scale):
        width = int(224 / scale)
        PpC = 8 / scale
        padded_im = np.pad(im, ((width, width),
                                (width, width), (0, 0)), mode='reflect')
        padded_hm = np.zeros(padded_im.shape[:-1])
        padded_im = padded_im.transpose((2, 0, 1))
        slider = ba.utils.sliding_window(
            padded_im, stride=int(width / 2), kernel_size=(width, width))
        full_hog = self.features(padded_im, feature_vector=False)
        for x1, y1, window in slider:
            # Image coordinate to hog coordniates:
            hx1 = int(max(x1 // PpC - 3 + 1, 0))
            hy1 = int(max(y1 // PpC - 3 + 1, 0))

            if hx1 > full_hog.shape[0] - 26:
                continue
            if hy1 > full_hog.shape[1] - 26:
                continue

            window_features = full_hog[hx1:hx1 + 26, hy1:hy1 + 26, ...]
            window_features = window_features.ravel().reshape(1, -1)

            score = self.model.predict(window_features)
            padded_hm[x1:x1 + width, y1:y1 + width] += max(0, score)
        hm = padded_hm[width:-width, width:-width]
        return hm

    def test(self, middlestr=''):
        slicedict = ba.utils.load(self.slicefile)
        scoreboxes = {}
        for img_bn, slicelist in tqdm(slicedict.items()):
            im = self.imread(self.images + img_bn + '.jpg')
            hm = np.zeros(im.shape[:-1])
            for scale in [1, 2]:
                hm += self.test_single_scale(im, scale)
            regions, rscores = ba.eval.scoreToRegion(hm)
            scoreboxes.update({img_bn: {'region': regions, 'score': rscores}})
        for boxdict in scoreboxes.values():
            boxdict['region'] = boxdict['region'].tolist()
            boxdict['score'] = boxdict['score'].tolist()
        tstr = time.strftime('_%b%d_%H:%M.scores.yaml', time.localtime())
        ba.utils.save(self.out + middlestr + tstr, scoreboxes)

    def train(self, nsamples):
        imlist = self.pickImages(nsamples)
        flow = ba.utils.SamplesGenerator(
            self.slicefile,
            imlist,
            self.images,
            self.negatives,
            ppI=self.ppI,
            patch_size=self.patch_size,
            ext=self.ext,
            mean=self.mean).flow(1000)
        samples, labels = next(flow)
        features = [self.features(sample) for sample in samples]
        labels = [int(i) for i in labels]
        self.model.fit(features, labels)

    def run(self, nsamples):
        self.train(nsamples)
        self.test('_' + str(nsamples) + 'samples')
