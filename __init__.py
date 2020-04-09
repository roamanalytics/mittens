try:
  from mittens.mittens.tf_mittens import Mittens, GloVe
except ImportError:
  from mittens.mittens.np_mittens import Mittens, GloVe

__version__ = "0.2.2"
