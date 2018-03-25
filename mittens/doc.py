BASE_DOC = """Vectorized {{framework}} implementation of {model}.
    {{second}}
    The `fit` method expects to be provided with a co-occurrence matrix
    which should be computed elsewhere.

    See https://nlp.stanford.edu/pubs/glove.pdf for details of GloVe.
    The GloVe implementation and default parameter values are taken
    from this paper.

    References
    ----------
    [1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 
    2014. GloVe: Global Vectors for Word Representation
    
    [2] Nick Dingwall and Christopher Potts. 2018. Mittens: An Extension 
    of GloVe for Learning Domain-Specialized Representations
    
    Parameters
    ----------
    n : int (default: 100)
        The embedding dimension.
    {mittens_param}
    xmax : int (default: 100)
        Word pairs with frequency greater than this are given weight 1.0.

        Word pairs with frequency under this are given weight
        (c / xmax) ** alpha, where c is the co-occurence count
        (see the paper, eq. (9)).

    alpha : float (default: 0.75)
        Exponent in the weighting function (see [1]_, eq. (9)).

    learning_rate : float (default: 0.01)
        Learning rate used for the Adagrad optimizer.

    tol : float (default: 1e-4)
        Stopping criterion for the loss.

    max_iter : int (default: 100)
        Number of training epochs. Default: 100, as in [1]_.

    log_dir : None or str (default: None)
        If `None`, no logs are kept.
        If `str`, this should be a directory in which to store
        Tensorboard logs. Logs are in fact stored in subdirectories
        of this directory to easily keep track of multiple runs.

    log_subdir : None or str (default: None)
        Use this to keep track of experiments. If `None`, 1 + the
        number of existing subdirectories is used to avoid overwrites.

        If `log_dir` is None, this value is ignored.

    display_progress : int (default: 10)
        Frequency with which to update the progress bar.
        If 0, no progress bar is shown.

    test_mode : bool
        If True, initial parameters are stored as `W_start`, `C_start`,
        `bw_start` and `bc_start`.

    Attributes
    ----------
    errors : list
        Tracks loss from each iteration during training.

    """

MITTENS_PARAM_DESCRIPTION = """
    mittens : float (default: 0.1)
        Relative weight assigned to remaining close to the original
        embeddings. Setting to 0 means that representations are
        initialized with the original embeddings but there is no
        penalty for deviating from them. Large positive values will
        heavily penalize any deviation, so the statistics of the new
        co-occurrence matrix will be increasingly ignored. A value of
        1 is 'balanced': equal weight is given to the GloVe objective
        on the new co-occurrence matrix and to remaining close to the
        original embeddings. 0.1 (the default) is recommended. See [2]_.
"""
