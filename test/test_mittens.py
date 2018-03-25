"""test_mittens.py

Test Mittens and GloVe using both NumPy and TensorFlow (if available).

If TensorFlow is not installed, those tests are skipped. If it is,
all tests are run twice: first with NumPy and then with TensorFlow,
according to the `framework` fixture.

Tests use pytest: from the command line, run:

$ pytest PATH/TO/MITTENS/test/

Add a `-v` flag to get detailed output.

Author: Nick Dingwall
"""
import numpy as np
import pytest

import mittens.np_mittens as np_mittens
try:
    TENSORFLOW_INSTALLED = True
    import mittens.tf_mittens as tf_mittens
except ImportError:
    TENSORFLOW_INSTALLED = False
    tf_mittens = None

FRAMEWORK_TO_MODULE = {'np': np_mittens, 'tf': tf_mittens}


@pytest.fixture(scope="module", params=['np', 'tf'])
def framework(request):
    return request.param


def test_glove(framework):
    if not TENSORFLOW_INSTALLED and framework == 'tf':
        pytest.skip("Tensorflow not installed.")

    np.random.seed(42)
    corr = _run_glove(FRAMEWORK_TO_MODULE[framework].GloVe, max_iter=1000)
    assert corr > 0.4


def test_glove_initialization(framework):
    if not TENSORFLOW_INSTALLED and framework == 'tf':
        pytest.skip("Tensorflow not installed.")

    np.random.seed(42)
    corr = _run_glove(FRAMEWORK_TO_MODULE[framework].GloVe, max_iter=0)
    assert abs(corr) < 0.2


def test_mittens(framework):
    """Test that Mittens moves initial representations in the correct
    direction.
    """
    if not TENSORFLOW_INSTALLED and framework == 'tf':
        pytest.skip("Tensorflow not installed.")

    np.random.seed(42)
    embedding_dim = 10
    vocab = ['a', 'b', 'c', 'd', 'e']
    initial_embeddings = {v: np.random.normal(0, 1, size=embedding_dim)
                          for v in vocab}
    X = _make_word_word_matrix(len(vocab))
    true = X.ravel()
    mittens = FRAMEWORK_TO_MODULE[framework].Mittens(n=embedding_dim,
                                                     max_iter=50)

    post_G = mittens.fit(X, vocab=vocab,
                         initial_embedding_dict=initial_embeddings)
    pre_G = mittens.G_start

    pre_pred = pre_G.dot(pre_G.T).ravel()
    post_pred = post_G.dot(post_G.T).ravel()

    pre_corr = _get_correlation(true, pre_pred)
    post_corr = _get_correlation(true, post_pred)
    assert post_corr > pre_corr


def test_mittens_parameter(framework):
    """Test that a large Mittens parameter keeps learned representations
    closer to the original than a small Mittens parameter.
    """
    if not TENSORFLOW_INSTALLED and framework == 'tf':
        pytest.skip("Tensorflow not installed.")

    np.random.seed(42)
    embedding_dim = 50
    vocab = ['a', 'b', 'c', 'd', 'e']
    initial_embeddings = {v: np.random.normal(0, 1, size=embedding_dim)
                          for v in vocab}
    X = _make_word_word_matrix(len(vocab))
    diffs = dict()
    small = 0.001
    mid = 1
    big = 1000
    for m in [small, mid, big]:
        mittens = FRAMEWORK_TO_MODULE[framework].Mittens(n=embedding_dim,
                                                         max_iter=50,
                                                         mittens=m)
        G = mittens.fit(X, vocab=vocab,
                        initial_embedding_dict=initial_embeddings)
        original = mittens.G_start
        diffs[m] = np.linalg.norm(G - original)
    assert diffs[small] > diffs[mid]
    assert diffs[mid] > diffs[big]


def _make_word_word_matrix(n=50):
    """Returns a symmetric matrix where the entries are drawn from a
    Poisson distribution"""
    base = np.random.zipf(2, size=(n, n)) - 1
    return base + base.T


def _get_correlation(true, pred):
    """Check correlation for nonzero elements of 'true'"""
    nonzero = true > 0
    return np.corrcoef(np.log(true[nonzero]), pred[nonzero])[0][1]


def _run_glove(glove_implementation, w=50, n=200, max_iter=100):
    X = _make_word_word_matrix(w)
    glove = glove_implementation(n=n, max_iter=max_iter)
    G = glove.fit(X)
    pred = G.dot(G.T).ravel()
    true = X.ravel()
    return _get_correlation(true, pred)

