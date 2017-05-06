import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
import ba.utils
import sklearn.metrics
from glob import glob

SEQUENTIAL_CMAP = sns.cubehelix_palette(start=0.5, rot=0.2, as_cmap=True)


def plt_hm(hm):
    fig, ax = newfig(0.9)
    plt.axis('off')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(hm, cmap=SEQUENTIAL_CMAP)
    return fig


def figsize(scale):
    fig_width_pt = 395.452
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    return fig_size


def newfig(width=0.9):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename):
    plt.gcf().savefig('{}.pgf'.format(filename), dpi=200)
    plt.gcf().savefig('{}.pdf'.format(filename), dpi=200)


def _prepareImagePlot(image):
    '''Prepares a PyPlot figure with an image.

    Args:
        image (image): The image

    Returns:
        the figure
    '''
    fig, ax = newfig(0.9)
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
        _fig = _prepareImagePlot(image)
    else:
        _fig = fig
    plt.imshow(overlay, cmap='viridis', alpha=0.5, interpolation='none')
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
                                        fill=None, alpha=1, ec=color,
                                        label=label))
        if label != '':
            bbox_props = dict(boxstyle='square', fc='w', ec='w')
            ax.text(rect[3] - 3, rect[0] + 5, label, ha='right', va='top',
                    size='xx-small', bbox=bbox_props)
    if path:
        fig.savefig(path, pad_inches=0, dpi=fig.dpi)
        plt.close(fig)
    return fig


def plt_results_for_tag(tag, mode):
    ITER = '500'

    root = './data/results/' + tag + '_FCN_*/'
    results = '*iter_' + ITER + '*results.mp'

    sns.set_palette("Set2", 5)
    fig, ax = newfig()
    roots = glob(root)
    roots.sort()
    for path in tqdm(roots, desc=tag):
        sp = path[len('./data/results/' + tag + '_FCN_'):-1]

        hitted_labels = []
        pred_labels = []
        for rmppath in tqdm(glob(path + results), desc=sp):
            rmp = ba.utils.load(rmppath)
            hitted_labels.extend(rmp[b'hitted_labels'])
            pred_labels.extend(rmp[b'pred_labels'])

        if mode == 'AUC':
            auc = sklearn.metrics.roc_auc_score(hitted_labels, pred_labels)
            tqdm.write('{}; {}'.format(sp, auc))
        elif mode == 'PR':
            pr, rc, th = sklearn.metrics.precision_recall_curve(
                hitted_labels, pred_labels, pos_label=1)
            ax.plot(rc, pr, label=sp)
        elif mode == 'ROC':
            fpr, tpr, th = sklearn.metrics.roc_curve(
                hitted_labels, pred_labels, pos_label=1)
            ax.plot(fpr, tpr, label=sp)
    if mode == 'PR':
        plt.xlabel('recall')
        plt.ylabel('precision')
        ax.legend()
        savefig('./build/' + tag + '_precs_recs')
    elif mode == 'ROC':
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        ax.legend()
        savefig('./build/' + tag + '_roc')
