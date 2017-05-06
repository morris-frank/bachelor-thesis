import skimage
import sklearn.svm
import ba.utils
import random
import scipy.misc
import numpy as np


class Baseline(object):
    def __init__(self, tag):
        self.tag = tag
        self.model = sklearn.svm.SVC()
        self.ks = 224
        self.patch_size = (self.ks, self.ks)
        self.images = 'data/datasets/voc2010/JPEGImages/'
        self.negatives = 'data/tmp/var_neg/'
        self.slicefile = 'data/tmp/pascpart/' + self.tag + '/seg.yaml'
        self.splitfile = 'data/tmp/pascpart/' + self.tag + '.txt'
        with open(self.splitfile, 'r') as f:
            self.imlist = [l[:-1] for l in f.readlines() if l.strip()]

    def imread(self, path):
        im = scipy.misc.imread(path)
        im = np.array(im, dtype=np.float32)
        im = im[:, :, ::-1]
        return im

    def features(image):
        hog_array = skimage.feature.hog(image)
        return hog_array


class BaselineTrainer(Baseline):
    def __init__(self, tag):
        super().__init_(tag)
        self.ppI = 10

    def pickImages(self, nsamples):
        return random.sample(self.imlist, nsamples)

    def test_single(self, im):
        padded_im = np.pad(im, ((self.ks, self.ks),
                                (self.ks, self.ks), (0, 0)), mode='reflect')
        padded_hm = np.zeros(padded_im.shape[:-1])
        slider = ba.utils.sliding_window(
            padded_im, stride=30, kernel_size=(self.ks, self.ks))
        for x1, x2, window in slider:
            window_features = self.features(window)
            score = self.model.predict(window_features)
            padded_hm[x1:x1 + self.ks, x2:x2 + self.ks] += score
        hm = padded_hm[self.ks:-self.ks, self.ks:-self.ks]
        regions, rscores = ba.eval.scoreToRegion(hm, imshape)
        return regions, rscores

    def test(self):
        slicedict = ba.utils.load(self.slicefile)
        for img_bn, slicelist in slicedict.values():
            im = self.imread(self.images + img_bn + '.jpg')
            self.test_single(im)

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
            mean=self.mean).flow(5000)
        samples, labels = next(flow)
        features = [self.features(sample) for sample in samples]
        self.model.fit(features, labels)
