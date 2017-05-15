from ba import BA_ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
import ba.utils
import sklearn.metrics
import re
import datetime
from glob import glob

SEQUENTIAL_CMAP = sns.cubehelix_palette(100, start=2.1, rot=-0.2, gamma=0.6,
                                        as_cmap=True)


def plt_hm(hm):
    fig, ax = newfig(0.9)
    plt.axis('off')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(hm, cmap=SEQUENTIAL_CMAP)
    return fig


def figsize(width, height):
    fig_width_pt = 395.452
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * width
    fig_height = fig_width * golden_mean * height
    fig_size = [fig_width, fig_height]
    return fig_size


def newfig(width=0.9, height=1.0):
    plt.clf()
    fig = plt.figure(figsize=figsize(width, height))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename):
    plt.gcf().savefig('{}.pgf'.format(filename), dpi=200)
    plt.gcf().savefig('{}.pdf'.format(filename), dpi=200)
    plt.clf()


def _prepareImagePlot(image):
    '''Prepares a PyPlot figure with an image.

    Args:
        image (image): The image

    Returns:
        the figure
    '''
    aspect_ratio = image.shape[0] / image.shape[1]
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig, ax = newfig(0.9, 0.9 * aspect_ratio / golden_mean)
    plt.axis('off')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image, interpolation='none')
    return fig, ax


def apply_overlay(image, overlay, path=None, label='', fig=None):
    '''Overlay overlay onto image and add label as text
    and save to path (full path with extension!)

    Args:
        image (image): The image to use as 'background'.
        overlay (image): The image to overly over the image.
        path (str, optional): The path to save the result to.
        label (str, optional): A label for the heatmap.
        fig (plt.figure, optional): An optional figure to work on
    '''
    if path is None and fig is None:
        raise RuntimeError
    if fig is None:
        _fig, ax = _prepareImagePlot(image)
    else:
        _fig = fig
    # overlay = overlay[..., np.newaxis]
    plt.imshow(overlay,
               cmap='binary_r', alpha=0.7, interpolation='bilinear')
    if label != '':
        patch = mpatches.Patch(color='yellow', label=label)
        plt.legend(handles=[patch])
    if fig is None:
        _fig.savefig(path, pad_inches=0, dpi=_fig.dpi)
        plt.close(_fig)
    return _fig


def apply_rect(image, rects, path=None, colors='black', labels=''):
    '''Overlay rectangle onto image and save to path
    (full path with extension!)

    Args:
        image (image): The image to use as 'background'.
        rects (tuple, list[tuple]): (xmin, ymin, xmax, ymax)
        path (str, optional): The full path to save the result to.
        color (str, list[str], optional): The color for the rectangles
        labels (str, list[str], optional): The for the rectangles
        fig (plt.figure, optional): An optional figure to work on
    '''
    fig, ax = _prepareImagePlot(image)
    if not isinstance(rects, list):
        rects = [rects]
    if not isinstance(colors, list):
        colors = [colors]
    if len(colors) < len(rects):
        colors = [colors[0]] * len(rects)
    if not isinstance(labels, list):
        labels = [labels]
    if len(labels) < len(rects):
        labels = [labels[0]] * len(rects)
    for rect, color, label in zip(rects, colors, labels):
        height = rect[2] - rect[0]
        width = rect[3] - rect[1]
        ax.add_patch(mpatches.Rectangle((rect[1], rect[0]), width, height,
                                        fill=None, alpha=1, ec=color, lw=2.5,
                                        label=label))
        if label != '':
            bbox_props = dict(boxstyle='square', fc='w', ec='w')
            ax.text(rect[3] - 3, rect[0] + 5, label, ha='right', va='top',
                    size='xx-small', bbox=bbox_props)
    if path:
        fig.savefig(path, pad_inches=0, dpi=fig.dpi)
        plt.close(fig)
    return fig, ax


def _plt_results(tag, mode, results_glob, nsamples, ax=None, startdate=None,
                 enddate=None):
    hitted_labels = []
    pred_labels = []
    auc = []
    datere = 'May([0-9]{2})_([0-9]{2}):([0-9]{2})'
    for rmppath in tqdm(glob(results_glob), desc=str(nsamples)):
        if startdate is not None or enddate is not None:
            day, _, _ = re.findall(datere, rmppath)[0]
            day = int(day)
            exec_date = datetime.date(2017, 5, day)
            if startdate is not None and exec_date < startdate:
                continue
            if enddate is not None and exec_date > enddate:
                continue
        hitted_labels = []
        pred_labels = []
        rmp = ba.utils.load(rmppath)
        hitted_labels.extend(rmp[b'hitted_labels'])
        pred_labels.extend(rmp[b'pred_labels'])
        auc.append(sklearn.metrics.roc_auc_score(hitted_labels, pred_labels))
    if len(hitted_labels) == 0:
        return False
    if mode == 'AUC':
        auc = sklearn.metrics.roc_auc_score(hitted_labels, pred_labels)
        # tqdm.write('{} with {}: {}'.format(tag, nsamples, auc))
        return auc
        # return np.var(auc)
    elif mode == 'PR':
        pr, rc, th = sklearn.metrics.precision_recall_curve(
            hitted_labels, pred_labels, pos_label=1)
        ax.plot(rc, pr, label=nsamples)
    elif mode == 'ROC':
        fpr, tpr, th = sklearn.metrics.roc_curve(
            hitted_labels, pred_labels, pos_label=1)
        ax.plot(fpr, tpr, label=nsamples)


def plt_results_for_tag(tag, mode, startdate=None, enddate=None):
    ITER = '500'

    root = 'data/results/' + tag + '_FCN_*/'
    results = '*iter_' + ITER + '*results.mp'
    sns.set_context('paper')
    sns.set_palette('Set2', 5)
    ax = None
    if mode != 'AUC':
        fig, ax = newfig()
    roots = glob(root)
    nsamples = [int(p[len('data/results/' + tag + '_FCN_'):-8])
                for p in roots]
    nsamples, roots = zip(*sorted(zip(nsamples, roots)))
    any_good = False
    auc_ret = {}
    for n, path in zip(tqdm(nsamples, desc=tag), roots):
        pltres = _plt_results(tag, mode, path + results, n, ax=ax,
                              startdate=startdate, enddate=enddate)
        if pltres is not False:
            any_good = True
            auc_ret[str(n)] = pltres
    if not any_good:
        return False
    if mode == 'PR':
        plt.title(r'\texttt{' + tag.replace('_', '\_') + '}')
        plt.xlabel('recall')
        plt.ylabel('precision')
        l = ax.legend(loc=1)
        l.set_title('samples')
        savefig('build/' + tag + '_precs_recs')
    elif mode == 'ROC':
        plt.title(r'\texttt{' + tag.replace('_', '\_') + '}')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        l = ax.legend(loc=4)
        l.set_title('samples')
        savefig('build/' + tag + '_roc')
    elif mode == 'AUC':
        return auc_ret
