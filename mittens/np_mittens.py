"""np_mittens.py

Fast implementations of Mittens and GloVe in Numpy.

See https://nlp.stanford.edu/pubs/glove.pdf for details of GloVe.

References
----------
[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning.
2014. GloVe: Global Vectors for Word Representation

[2] Nick Dingwall and Christopher Potts. 2018. Mittens: An Extension
of GloVe for Learning Domain-Specialized Representations

Authors: Nick Dingwall, Chris Potts
"""
import numpy as np

from mittens.mittens_base import randmatrix, noise
from mittens.mittens_base import MittensBase, GloVeBase


_FRAMEWORK = "NumPy"
_DESC = """
    The TensorFlow version is faster, especially if used on GPU. 
    To use it, install TensorFlow, restart your Python kernel and 
    import from the base class:

    >>> from mittens import {model}
    """


class Mittens(MittensBase):
    __doc__ = MittensBase.__doc__.format(
        framework=_FRAMEWORK,
        second=_DESC.format(model=MittensBase._MODEL))

    @property
    def framework(self):
        return _FRAMEWORK

    def _fit(self, coincidence, weights, log_coincidence,
             vocab=None,
             initial_embedding_dict=None,
             fixed_initialization=None):
        self._initialize_w_c_b(self.n_words, vocab, initial_embedding_dict)

        if fixed_initialization is not None:
            assert self.test_mode
            self.W = fixed_initialization['W']
            self.C = fixed_initialization['C']
            self.bw = fixed_initialization['bw']
            self.bc = fixed_initialization['bc']

        if self.test_mode:
            # These are stored for testing
            self.W_start = self.W.copy()
            self.C_start = self.C.copy()
            self.bw_start = self.bw.copy()
            self.bc_start = self.bc.copy()

        for iteration in range(self.max_iter):
            pred = self._make_prediction()
            gradients, error = self._get_gradients_and_error(
                pred, log_coincidence, weights)
            self._check_shapes(gradients)
            self.errors.append(error)
            self._apply_updates(gradients)
            self._progressbar("error {:4.4f}".format(error), iteration)
        return self.W + self.C

    def _check_shapes(self, gradients):
        assert gradients['W'].shape == self.W.shape
        assert gradients['C'].shape == self.C.shape
        assert gradients['bw'].shape == self.bw.shape
        assert gradients['bc'].shape == self.bc.shape

    def _initialize_w_c_b(self, n_words, vocab, initial_embedding_dict):
        self.W = randmatrix(n_words, self.n)  # Word weights.
        self.C = randmatrix(n_words, self.n)  # Context weights.
        if initial_embedding_dict:
            assert self.n == len(next(iter(initial_embedding_dict.values())))

            self.original_embedding = np.zeros((len(vocab), self.n))
            self.has_embedding = np.zeros(len(vocab), dtype=bool)

            for i, w in enumerate(vocab):
                if w in initial_embedding_dict:
                    self.has_embedding[i] = 1
                    embedding = np.array(initial_embedding_dict[w])
                    self.original_embedding[i] = embedding
                    # Divide the original embedding into W and C,
                    # plus some noise to break the symmetry that would
                    # otherwise cause both gradient updates to be
                    # identical.
                    self.W[i] = 0.5 * embedding + noise(self.n)
                    self.C[i] = 0.5 * embedding + noise(self.n)
            # This is for testing. It differs from
            # `self.original_embedding` only in that it includes the
            # random noise we added above to break the symmetry.
            self.G_start = self.W + self.C

        self.bw = randmatrix(n_words, 1)
        self.bc = randmatrix(n_words, 1)
        self.ones = np.ones((n_words, 1))

    def _make_prediction(self):
        # Here we make use of numpy's broadcasting rules
        pred = np.dot(self.W, self.C.T) + self.bw + self.bc.T
        return pred

    def _get_gradients_and_error(self,
                                 predictions,
                                 log_coincidence,
                                 weights):
        # First we compute the GloVe gradients
        diffs = predictions - log_coincidence
        weighted_diffs = np.multiply(weights, diffs)
        wgrad = weighted_diffs.dot(self.C)
        cgrad = weighted_diffs.T.dot(self.W)
        bwgrad = weighted_diffs.sum(axis=1).reshape(-1, 1)
        bcgrad = weighted_diffs.sum(axis=0).reshape(-1, 1)
        error = (0.5 * np.multiply(weights, diffs ** 2)).sum()

        # Then we add the Mittens term (only if mittens > 0)
        if self.mittens > 0:
            curr_embedding = self.W + self.C
            distance = curr_embedding[self.has_embedding, :] - \
                       self.original_embedding[self.has_embedding, :]
            wgrad[self.has_embedding, :] += 2 * self.mittens * distance
            cgrad[self.has_embedding, :] += 2 * self.mittens * distance
            error += self.mittens * (
                np.linalg.norm(distance, ord=2, axis=1) ** 2).sum()
        return {'W': wgrad, 'C': cgrad, 'bw': bwgrad, 'bc': bcgrad}, error

    def _apply_updates(self, gradients):
        """Apply AdaGrad update to parameters.

        Parameters
        ----------
        gradients

        Returns
        -------

        """
        if not hasattr(self, 'optimizers'):
            self.optimizers = \
                {obj: AdaGradOptimizer(self.learning_rate)
                 for obj in ['W', 'C', 'bw', 'bc']}
        self.W -= self.optimizers['W'].get_step(gradients['W'])
        self.C -= self.optimizers['C'].get_step(gradients['C'])
        self.bw -= self.optimizers['bw'].get_step(gradients['bw'])
        self.bc -= self.optimizers['bc'].get_step(gradients['bc'])


class AdaGradOptimizer:
    """Simple AdaGrad optimizer.

    This is loosely based on the Tensorflow version. See
    https://github.com/tensorflow/tensorflow/blob/master/
    tensorflow/python/training/adagrad.py.

    Parameters
    ----------
    learning_rate : float
    initial_accumulator_value : float (default: 0.1)
        Initialize the momentum with this value.
    """

    def __init__(self, learning_rate, initial_accumulator_value=0.1):
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self._momentum = None

    def get_step(self, grad):
        """Computes the 'step' to take for the next gradient descent update.

        Returns the step rather than performing the update so that
        parameters can be updated in place rather than overwritten.

        Examples
        --------
        >>> gradient = # ...
        >>> optimizer = AdaGradOptimizer(0.01)
        >>> params -= optimizer.get_step(gradient)

        Parameters
        ----------
        grad

        Returns
        -------
        np.array
            Size matches `grad`.
        """
        if self._momentum is None:
            self._momentum = self.initial_accumulator_value * np.ones_like(grad)
        self._momentum += grad ** 2
        return self.learning_rate * grad / np.sqrt(self._momentum)


class GloVe(GloVeBase, Mittens):

    __doc__ = GloVeBase.__doc__.format(
        framework=_FRAMEWORK,
        second=_DESC.format(model=GloVeBase._MODEL))
