<img src="img/mittens_logo.png" alt="title" width="100">

# Mittens

This package contains fast [TensorFlow](https://github.com/tensorflow/tensorflow) and [NumPy](https://github.com/numpy/numpy) implementations of [GloVe](https://nlp.stanford.edu/projects/glove/) and [Mittens](arvix_link.com).

By vectorizing the GloVe objective function, we deliver massive speed gains over other Python implementations (10x on CPU; 60x on GPU). See the [Speed](#speed) section below.

The caveat is that our implementation is only suitable for modest vocabularies (up to ~20k tokens should be fine) since the co-occurrence matrix must be held in memory.

Vectorizing the objective also reveals that it is amenable to a retrofitting term that encourages representations to remain close to pretrained embeddings. This is useful for domains that require specialized representations but lack sufficient data to train them from scratch. Mittens starts with the general-purpose pretrained representations and tunes them to a specialized domain.

## Installation

### Dependencies

Mittens only requires `numpy`. However, if `tensorflow` is available, that will be used instead. The two implementations use the same cost function and optimizer, so the only difference is that the `tensorflow` version shows a small speed improvement on CPU, and a large speed improvement when run on GPU.

### User installation

The easiest way to install `mittens` is with `pip`:

```
pip install -U mittens
```

You can also install it by cloning the repository and adding it to your Python path. Make sure you have at least `numpy` installed.

Note that neither method automatically installs TensorFlow: see [their instructions](https://www.tensorflow.org/install/).


## Examples

For both examples, it is assumed that you have already computed the weighted co-occurrence matrix (`cooccurence` for vocabulary `vocab`).

### GloVe

```
from mittens import GloVe

# Load `cooccurrence`
# Train GloVe model
glove_model = GloVe(n=25, max_iter=1000)  # 25 is the embedding dimension
embeddings = glove_model.fit(cooccurrence)
```

`embeddings` is now an `np.array` of size `(len(vocab), n)`, where the rows correspond to the tokens in `vocab`.

A small complete example:

```
from mittens import GloVe
import numpy as np

cooccurrence = np.array([
    [  4.,   4.,   2.,   0.],
    [  4.,  61.,   8.,  18.],
    [  2.,   8.,  10.,   0.],
    [  0.,  18.,   0.,   5.]])
glove_model = GloVe(n=2, max_iter=100)
embeddings = glove_model.fit(cooccurrence)
embeddings

array([[ 1.13700831, -1.16577291],
       [ 2.52644205,  1.56363213],
       [ 0.2376546 ,  0.96793109],
       [ 0.41685158,  1.32988596]], dtype=float32)
```

### Mittens

To use Mittens, you first need pre-trained embeddings. In our paper, we used Pennington et al's embeddings, available from the [Stanford GloVe website](https://nlp.stanford.edu/projects/glove/).

These vectors should be stored in a dict, where the key is the token and the value is the vector. For example, the function `glove2dict` below manipulates a Stanford embedding file into the appropriate format.

```
import csv
import numpy as np

def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

```

Now that we have our embeddings (stored as `original_embeddings`), as well as a co-occurrence matrix and associated vocabulary, we're ready to train Mittens:

```
from mittens import Mittens

# Load `cooccurrence` and `vocab`
# Load `original_embedding`
mittens_model = Mittens(n=50, max_iter=1000)
# Note: n must match the original embedding dimension
new_embeddings = mittens_mode.fit(
    cooccurrence,
    vocab=vocab,
    initial_embedding_dict= original_embedding)
```

Once trained, `new_embeddings` should be *compatible* with the existing embeddings in the sense that they will be oriented such that using a mix of the the two embeddings is meaningful (e.g. using original embeddings for any test-set tokens that were not in the training set).


## <a name="speed"></a>Speed

We compared the per-epoch speed (measured in seconds) for a variety of vocabulary sizes using randomly-generated co-occurrence matrices that were approximately 90% sparse. As we see here, for matrices that fit into memory, performance is competitive with the [official C implementation](https://github.com/stanfordnlp/GloVe) when run on a GPU.

For denser co-occurrence matrices, Mittens will have an advantage over the C implementation since it's speed does not depend on sparsity, while the official release is linear in the number of non-zero entries.

|                           | 5K (CPU) | 10K (CPU) | 20K (CPU) | 5K (GPU) | 10K (GPU) | 20K (GPU) |
|:--------------------------|---------:|----------:|----------:|---------:|----------:|----------:|
| Non-vectorized TensorFlow |     14.02|      63.80|     252.65|     13.56|      55.51|     226.41|
| Vectorized Numpy          |      1.48|       7.35|      50.03|         −|          −|          −|
| Vectorized TensorFlow     |      1.19|       5.00|      28.69|      0.27|       0.95|       3.68|
| Official GloVe            |      0.66|       1.24|       3.50|         −|          −|          −|

## References
[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. *GloVe: Global Vectors for Word Representation*.

[2] Nicholas Dingwall and Christopher Potts. 2018. *Mittens: An Extension of GloVe for Learning Domain-Specialized Representations*. (NAACL 2018)
