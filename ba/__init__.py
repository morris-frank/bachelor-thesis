BA_ROOT = '/net/hciserver03/storage/mfrank/src/ba/'
from ba import utils
from ba.set import SetList
from ba.pascalpart import PascalPart
from ba.pascalpart import PascalPartSet

import seaborn as sns
import matplotlib as mpl
sns.set_style('whitegrid')

# mpl.use('pgf')
pgf_with_latex = {
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
    'text.usetex': True,
    'pgf.preamble': [
        r'\usepackage[utf8x]{inputenc}',
        r'\usepackage[T1]{fontenc}',
        ],
    'figure.autolayout': True
    }
mpl.rcParams.update(pgf_with_latex)
