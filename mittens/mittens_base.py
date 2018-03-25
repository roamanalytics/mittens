from copy import copy
import random
import sys

import numpy as np

from mittens.doc import BASE_DOC, MITTENS_PARAM_DESCRIPTION


class MittensBase(object):

    _MODEL = "Mittens"
    __doc__ = BASE_DOC.format(model=_MODEL,
                              mittens_param=MITTENS_PARAM_DESCRIPTION)

    def __init__(self, n=100, mittens=0.1, xmax=100, alpha=0.75,
                 max_iter=100, learning_rate=0.05, tol=1e-4,
                 display_progress=10, log_dir=None, log_subdir=None,
                 test_mode=False, **kwargs):
        self.n = n
        self.mittens = float(mittens)
        self.xmax = xmax
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.tol = tol
        self.display_progress = display_progress
        self.model = None
        self.n_words = None
        self.log_dir = log_dir
        self.log_subdir = log_subdir
        self.max_iter = max_iter
        self.errors = list()
        self.test_mode = test_mode

    def fit(self,
            X,
            vocab=None,
            initial_embedding_dict=None,
            fixed_initialization=None):
        """Run GloVe and return the new matrix.

        Parameters
        ----------
        X : array-like of shape = [n_words, n_words]
            The square count matrix.

        vocab : iterable or None (default: None)
            Rownames for `X`.

        initial_embedding_dict : dict or None (default: None)
            Map into representations that we want to use for a "warm
            start" -- e.g., GloVe vectors trained on a massive corpus.

            Learned representations of words in `vocab` are initialized
            from `initial_embedding_dict` wherever possible; words not
            found in `initial_embedding_dict` are randomly initialized.

        fixed_initialization : dict or None (default: None)
            If a dict, this will replace the random initializations
            of `W`, `C`, `bw` and `bc`. Dict keys must be
            ['W', 'C', 'bw', 'bc'], and values should be np.arrays
            of appropriate size ((n_words, n) for `W` and `C`, and
            (n_words, ) for `bw` and `bc`).

        Returns
        -------
        np.array
            Of shape = [n_words, embedding_dim]. I.e. each row is the
            embedding of the corresponding element in `vocab`.

        """
        if fixed_initialization is not None:
            assert self.test_mode, \
                "Fixed initialization parameters can only be provided" \
                " in test mode. Initialize {} with `test_mode=True`.". \
                     format(self.__class__.split(".")[-1])
        self._check_dimensions(
            X, vocab, initial_embedding_dict
        )
        weights, log_coincidence = self._initialize(X)
        return self._fit(X, weights, log_coincidence,
                         vocab=vocab,
                         initial_embedding_dict=initial_embedding_dict,
                         fixed_initialization=fixed_initialization)

    @property
    def framework(self):
        """Indicates the framework used for model training (either
        "NumPy" or "TensorFlow").
        """
        raise NotImplementedError

    def _fit(self, coincidence, weights, log_coincidence,
             vocab=None,
             initial_embedding_dict=None,
             fixed_initialization=None):
        """Called by self.fit and implemented by subclasses.

        Should return a np.array whose first dimension matches
        `coincidence` (the co-occurrence matrix) and whose second
        is equal to the dimension of the embedding (`self.n`).

        Parameters
        ----------
        coincidence : np.array (n_words, n_words)

        weights : np.array (n_words, n_words)
            Corresponds to the function `f` in [1]_.

        log_coincidence : np.array (n_words, n_words)
            Log of non-zero co-occurrence counts. Corresponds to
            function `g` in [2]_.

        vocab : iterable or None

        initial_embedding_dict : dict or None

        fixed_initialization : dict or None

        Returns
        -------
        np.array
            Of shape = [n_words, embedding_dim].
        """
        raise NotImplementedError

    def _check_dimensions(self, X, vocab, initial_embedding_dict):
        if vocab:
            assert X.shape[0] == len(vocab), \
                "Vocab has {} tokens, but expected {} " \
                "(since X has shape {}).".format(
                    len(vocab), X.shape[0], X.shape)

        if initial_embedding_dict:
            embeddings = initial_embedding_dict.values()
            sample_len = len(random.choice(list(embeddings)))
            assert sample_len == self.n, \
                "Initial embedding contains {}-dimensional embeddings," \
                " but {}-dimensional were expected.".\
                    format(sample_len, self.n)

    def _initialize(self, coincidence):
        self.n_words = coincidence.shape[0]
        bounded = np.minimum(coincidence, self.xmax)
        weights = (bounded / float(self.xmax)) ** self.alpha
        log_coincidence = log_of_array_ignoring_zeros(coincidence)
        return weights, log_coincidence

    def _progressbar(self, msg, iter_num):
        """Display a progress bar with current loss.

        Parameters
        ----------
        msg : str
            Message to print alongside the progress bar

        iter_num : int
            Iteration number.
            Progress is only printed if this is a multiple of
            `self.display_progress`.

        """
        if self.display_progress and \
                                (iter_num + 1) % self.display_progress == 0:
            sys.stderr.write('\r')
            sys.stderr.write("Iteration {}: {}".format(iter_num + 1, msg))
            sys.stderr.flush()

    def __repr__(self):
        params = copy(self.__dict__)
        params.pop("errors")
        params['framework'] = self.framework
        inset = len(self._MODEL) + 1
        max_width = 72 - inset
        parts = [_format_param_value(k, v) for k, v in sorted(params.items())]
        param_str = ""
        for p in parts:
            if len(param_str) == 0:
                sep = ""
            elif len(param_str.split("\n")[-1]) + len(p) + 4 > max_width:
                sep = ",\n" + " " * inset
            else:
                sep = ", "
            param_str += sep + p
        return "{}({})".format(self._MODEL, param_str)


def _format_param_value(key, value):
    """Wraps string values in quotes, and returns as 'key=value'.
    """
    if isinstance(value, str):
        value = "'{}'".format(value)
    return "{}={}".format(key, value)


class GloVeBase(MittensBase):

    _MODEL = "GloVe"
    __doc__ = BASE_DOC.format(model=_MODEL,
                              mittens_param="")

    def __init__(self, n=100, xmax=100, alpha=0.75, max_iter=100,
                 learning_rate=0.05, tol=1e-4, display_progress=10,
                 log_dir=None, log_subdir=None, test_mode=False):
        super(GloVeBase, self).__init__(n=n,
                                        xmax=xmax,
                                        alpha=alpha,
                                        mittens=0,
                                        max_iter=max_iter,
                                        learning_rate=learning_rate,
                                        tol=tol,
                                        display_progress=display_progress,
                                        log_dir=log_dir,
                                        log_subdir=log_subdir,
                                        test_mode=test_mode)

    def fit(self, X, fixed_initialization=None):
        """Run GloVe and return the new matrix.

        Parameters
        ----------
        X : array-like of shape = [n_words, n_words]
            The square count matrix.

        fixed_initialization : None or dict (default: None)
            If a dict, this will replace the random initializations
            of `W`, `C`, `bw` and `bc`. Dict keys must be
            ['W', 'C', 'bw', 'bc'], and values should be np.arrays
            of appropriate size ((n_words, n) for `W` and `C`, and
            (n_words, ) for `bw` and `bc`).

        Returns
        -------
        np.array
            The learned vectors, of dimensionality `(len(X), self.n)`
            I.e. each row is the embedding of the corresponding element
            of X.
        """
        return super(GloVeBase, self).fit(
            X, fixed_initialization=fixed_initialization)


def log_of_array_ignoring_zeros(M):
    """Returns an array containing the logs of the nonzero
    elements of M. Zeros are left alone since log(0) isn't
    defined.

    Parameters
    ----------
    M : array-like

    Returns
    -------
    array-like
        Shape matches `M`

    """
    log_M = M.copy()
    mask = log_M > 0
    log_M[mask] = np.log(log_M[mask])
    return log_M


def randmatrix(m, n, random_seed=None):
    """Creates an m x n matrix of random values drawn using
    the Xavier Glorot method."""
    val = np.sqrt(6.0 / (m + n))
    np.random.seed(random_seed)
    return np.random.uniform(-val, val, size=(m, n))


def noise(n, scale=0.01):
    """Sample zero-mean Gaussian-distributed noise.

    Parameters
    ----------
    n : int
        Number of samples to take

    scale : float
        Standard deviation of the noise.

    Returns
    -------
    np.array of size (n, )
    """
    return np.random.normal(0, scale, size=n)
