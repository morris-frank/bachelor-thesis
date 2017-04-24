from ba import caffeine
from ba import utils
from ba import netrunner
from ba.set import SetList
from ba.pascalpart import PascalPart
from ba.pascalpart import PascalPartSet

import seaborn as sns
import matplotlib as mpl
sns.set_style('whitegrid')
# mpl.use('pgf')
pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        ]
    }
mpl.rcParams.update(pgf_with_latex)
