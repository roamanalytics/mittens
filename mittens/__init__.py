try:
    from mittens.tf_mittens import Mittens, GloVe
except ImportError:
    from mittens.np_mittens import Mittens, GloVe

__version__ = "0.1"
