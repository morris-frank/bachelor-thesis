# For the pure python implementation:
from ba.selectivesearch import selective_search as selective_search_py
import ba.utils
from matplotlib import pyplot as plt
import numpy as np
import skimage.transform
from scipy.ndimage import distance_transform_edt as euclidean_distance_transform
from scipy.misc import imread, imresize
# For the cpython implementation:
from selective_search import selective_search as selective_search_cpy
import sys
from tqdm import tqdm
import yaml


def evalYAML(predf, gtf, images, heatmaps=None):
    '''Evaluates the predicted regions from a test run. Writes the performances
    to an other YAML file.

    Args:
        predf (str): The path to the YAML file containing the predictions
        gtf (str): The path to the YAML file containing ground truth regions
        images (str): The path to the original images
        heatmaps (str, optional): The path to the original heatmaps
    '''
    preds = ba.utils.loadYAML(predf)
    gts = ba.utils.loadYAML(gtf)
    outputfile = '.'.join(predf.split('.')[:-2] + ['evals', 'yaml'])
    outputdir = ba.utils.touch('.'.join(predf.split('.')[:-2]) + '/evals/')
    results = {}
    ext_img = ba.utils.prevalentExtension(images)
    if heatmaps is not None:
        ext_hm = ba.utils.prevalentExtension(heatmaps)
    print('Evaluating {}'.format(predf))
    for idx, pred in tqdm(preds.items()):
        rect = pred['region']
        score = pred['score']
        im = imread('{}{}.{}'.format(images, idx, ext_img))
        if heatmaps is not None:
            hm = imread('{}{}.{}'.format(heatmaps, idx, ext_hm))
            hm = imresize(hm, im.shape[:-1])
        imout = outputdir + idx + '.png'
        # Get the ground truth:
        gtslice = gts[idx]
        gtrect = (gtslice[0].start, gtslice[1].start,
                  gtslice[0].stop, gtslice[1].stop)

        # Evaluate it:
        iOU = intersectOverUnion(rect, gtrect)
        l2dist = rectDistance(rect, gtrect)
        diagonal = np.linalg.norm(im.shape[:-1])
        disterr = float(l2dist / diagonal)
        scalingerr = float(np.linalg.norm(
            [(rect[2] - rect[0]) / (gtrect[2] - gtrect[0]),
             (rect[3] - rect[1]) / (gtrect[3] - gtrect[1])]))
        results[idx] = {'iOU': iOU, 'disterr': disterr,
                        'scalingerr': scalingerr}

        # Save overlay image:
        if heatmaps is None:
            ba.utils.apply_rect(im, [rect, gtrect], imout, ['red', 'green'],
                                [score, ''])
        else:
            fig = ba.utils._prepareImagePlot(im)
            ba.utils.apply_overlay(im, hm, imout, fig=fig)
            ba.utils.apply_rect(im, [rect, gtrect], imout, ['red', 'green'],
                                [score, ''], fig=fig)
            fig.savefig(imout, pad_inches=0, dpi=fig.dpi)
            plt.close(fig)

    meanIOU = float(np.mean([i['iOU'] for i in results.values()]))
    meanDistErr = float(np.mean([i['disterr'] for i in results.values()]))
    meanScalErr = float(np.mean([i['scalingerr'] for i in results.values()]))

    results['mean_iOU'] = meanIOU
    results['mean_disterr'] = meanDistErr
    results['mean_scalingerr'] = meanScalErr

    with open(outputfile, 'w') as f:
        yaml.dump(results, f)
    return meanIOU, meanDistErr, meanScalErr, len(results)


def rectDistance(a, b):
    '''Calculates the distance between the centers of two rectangles.

    Args:
        a (tuple): (xmin ymin xmax ymax)
        b (tuple): (xmin ymin xmax ymax)

    Returns the iOU
    '''
    aCenter = np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0])
    bCenter = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
    return np.linalg.norm(aCenter - bCenter)


def intersectOverUnion(a, b):
    '''Calculates the intersect over Union of two rectangles.

    Args:
        a (tuple): (xmin ymin xmax ymax)
        b (tuple): (xmin ymin xmax ymax)

    Returns the iOU
    '''
    area = intersectArea(a, b)
    if area == 0:
        return area
    aArea = (a[2] - a[0]) * (a[3] - a[1])
    bArea = (b[2] - b[0]) * (b[3] - b[1])
    iOU = area / float(aArea + bArea - area)
    return iOU


def intersectArea(a, b):
    '''Calculates the area of the intersecting area of two rectangles.

    Args:
        a (tuple): (xmin ymin xmax ymax)
        b (tuple): (xmin ymin xmax ymax)

    Returns the area
    '''
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx >= 0 and dy >= 0:
        return dx * dy
    else:
        return 0.0


def _selective_search(image, CPY=True):
    '''Selective search meta function.

    Args:
        image (ndarray): The image
        CPY (bool, optional): What version to use

    Returns:
        the Regions
    '''
    if CPY:
        # cpython implementation
        mshape = max(image.shape)
        ks = [int(mshape/i) for i in [10, 7, 5, 3]]
        regions = selective_search_cpy(image, ks=ks, n_jobs=-10)
    else:
        # pure python implementation
        regions = []
        labels, regions_ = selective_search_py(
            image, scale=500, sigma=0.9, min_size=200)
        for region in regions_:
            y1, x1, y2, x2 = region['rect']
            regions.append((region['size'], (x1, y1, x2, y2)))

    return regions


def scoreToRegion(heatmap, image):
    '''Reudces a heatmap to a bounding box by searching through regions of the
    image generated by an selective search.

    Args:
        heatmap (ndarray): The map of un-normalized scores
        image (ndarray): The image

    Returns:
        The maximum bounding box (x_start, y_start, x_end, y_end)
    '''
    regions = _selective_search(image, True)

    # Add all found regions to the start and end lists:
    starts = []
    ends = []
    minsize = int(heatmap.size * 0.01)
    for size, (x1, y1, x2, y2) in regions:
        if x2 < x1:
            x2,x1 = x1,x2
        if y2 < y1:
            y2,y1 = y1,y2
        if (x2 - x1) * (y2 - y1) > minsize:
            starts.append((x1, y1))
            ends.append((x2, y2))

    heatmap_sum = float(np.sum(heatmap))
    if heatmap_sum > 0:
        heatmap /= heatmap_sum

    # Add distance base negative penalty:
    thres = 0.1 / heatmap_sum
    negative_heatmap = euclidean_distance_transform(heatmap < thres)
    negative_heatmap /= np.sum(negative_heatmap)
    heatmap -= negative_heatmap

    # Get the sums in all regions:
    integral_scores = skimage.transform.integral_image(heatmap)
    bbscores = skimage.transform.integrate(integral_scores, starts, ends)

    # Get the score density per region:
    for idx, (start, end) in enumerate(zip(starts, ends)):
        area = (end[0] - start[0]) * (end[1] - start[1])
        if area > 0:
            bbscores[idx] /= area

    # Get the maximum performing region:
    bbidx = bbscores.argmax()
    x, y = starts[bbidx]
    x2, y2 = ends[bbidx]
    regmax = (int(x), int(y), int(x2), int(y2))

    return regmax, bbscores[bbidx]
