try:
  try:
    from mittens.tf_mittens import Mittens, GloVe
  except:
#    print("Failed mittens.tf_mittens")
    from mittens.mittens.tf_mittens import Mittens, GloVe
except ImportError:
#  print("Failed ANY tf_mittens")
  try:
    from mittens.np_mittens import Mittens, GloVe
  except:
#    print("Failed mittens.np_mittens")
    from mittens.mittens.np_mittens import Mittens, GloVe

__version__ = "0.2.2"
