import ba.utils
import ba.plt
import matplotlib.pyplot as plt
from glob import glob

ROOT = './data/results/*/*prec_rec.mp'

for path in glob(ROOT):
    content = ba.utils.load(path)
    fig, ax = ba.plt.newfig()
    ax.plot(content['recall'], content['precision'])
    plt.xlabel('recall')
    plt.xlabel('precision')
    ba.plt.savefig(path[:-2] + 'pgf')
    ba.plt.savefig(path[:-2] + 'png')
