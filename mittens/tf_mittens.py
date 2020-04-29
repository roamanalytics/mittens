"""tf_mittens.py

Fast implementations of Mittens and GloVe in Tensorflow.

See https://nlp.stanford.edu/pubs/glove.pdf for details of GloVe.

Authors: Nick Dingwall, Chris Potts
"""
import os

import numpy as np

# Try to accommodate TensorFlow v1 and v2:
try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
except ImportError:
    import tensorflow as tf

from time import time

try:
  from mittens.mittens_base import randmatrix, noise
  from mittens.mittens_base import MittensBase, GloVeBase
except:
  from mittens.mittens.mittens_base import randmatrix, noise
  from mittens.mittens.mittens_base import MittensBase, GloVeBase


from collections import deque

__VER__ = '0.2.1.6'


_FRAMEWORK = "TensorFlow"
_DESC = """
    This version is faster than the NumPy version. If you prefer NumPy
    for some reason, import it directly:

    >>> from mittens.np_mittens import {model}
    """


class Mittens(MittensBase):

    __doc__ = MittensBase.__doc__.format(
        framework=_FRAMEWORK,
        second=_DESC.format(model=MittensBase._MODEL))
    
    def __init__(self, 
                 DEBUG=False, 
                 no_feeds=True, 
                 save_folder=None,
                 save_iters=500,
                 save_opt_hist=True,
                 name='mittenstf',
                 **kwargs):
        super().__init__(**kwargs)
        self.DEBUG = DEBUG
        self.name = name
        self.save_iters = save_iters
        self.save_opt_hist = save_opt_hist
        self.no_feeds = no_feeds
        self.message("Tensorflow ({}) Mittens v{} initialized with {}".format(
            tf.__version__,
            __VER__,
            'full in-GPU training (no memory feeds)' if self.no_feeds else 'memory feeds'
            ))
        self.save_folder = ''
        if save_folder is not None:
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            if os.path.isdir(save_folder):
                self.save_folder = save_folder
            self.message("  Saving in '{}' folder.".format(self.save_folder))
        else:
            self.message('  No folder provided. Saving in current folder.')
            
        self.message("  Generating d={} embeddings for {} items".format(
            self.n,
            self.n_words))
          
        self._last_saved_file = None
        return

    @property
    def framework(self):
        return _FRAMEWORK
      
    
    def save(self, filename):
        fn = os.path.join(self.save_folder, filename)
        embeds = self._get_embeds()
        try:
          np.save(fn, embeds)
          self.message('')
          self.message("  Embeddings file '{}' saved.".format(fn))
          res = fn + '.npy'
        except:
          res = None
        return res
      
      
    def _save_status(self, itr):
        if self._last_saved_file is not None:
          try:
            os.remove(self._last_saved_file)
          except:
            self.message('')
            self.message("Could not remove '{}'".format(self._last_saved_file))
        fn = '{}_i{}k'.format(self.name, int(itr / 1000))
        self._last_saved_file = self.save(fn)
    
    def _save_optimization_history(self, skip=5):
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        _ = plt.figure()
        ax = plt.gca()
        ax.plot(np.arange(skip, len(self.errors)), self.errors[skip:])
        ax.set_title('Mittens loss history (skipped first {} iters)'.format(skip))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
#        ax.set_xscale('log')
        plt.savefig(os.path.join(self.save_folder, '{}_loss.png'.format(self.name)))
        plt.close()
        


    def _fit(self, X, weights, log_coincidence,
             vocab=None,
             initial_embedding_dict=None,
             fixed_initialization=None):
        if fixed_initialization is not None:
            raise AttributeError("Tensorflow version of Mittens does "
                                 "not support specifying initializations.")
        
        self.message("Preparing graph & session:", timer='start')

        # Start the session:
        if hasattr(self, 'sess'):
          self.sess.close()
          self.sess = None
        run_config = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        # Build the computation graph.
        self.message("  Building graph")
        self._build_graph(vocab, initial_embedding_dict)

        # Optimizer set-up:
        self.message("  Preparing cost/train function")
        if self.no_feeds:
            self.cost = self._get_cost_function(weights, log_coincidence)
        else:
            self.cost = self._get_cost_function_with_placeholders()
        self.optimizer = self._get_train_func()

        # Set up logging for Tensorboard
        if self.log_dir:
            n_subdirs = len(os.listdir(self.log_dir))
            subdir = self.log_subdir or str(n_subdirs + 1)
            directory = os.path.join(self.log_dir, subdir)
            log_writer = tf.summary.FileWriter(directory, flush_secs=1)

        # Run training
        self.message("  Initializing variables")
        self.sess.run(tf.global_variables_initializer())
        if self.test_mode:
            self.W_start = self.sess.run(self.W)
            self.C_start = self.sess.run(self.C)
            self.bw_start = self.sess.run(self.bw)
            self.bc_start = self.sess.run(self.bc)

        merged_logs = tf.summary.merge_all()
        t0 = time()
        self._last_timings = deque(maxlen=1000)
        self.message("Done preparation session", timer='stop')
        for i in range(1, self.max_iter+1):
            t1 = time()
            if not self.no_feeds:
                feed_dict = {
                    self.weights: weights,
                    self.log_coincidence: log_coincidence
                    }
            else:
                feed_dict = None
              
            _, loss, stats = self.sess.run(
                [self.optimizer, self.cost, merged_logs],
                feed_dict=feed_dict,
                options=run_config,
                )

            # Keep track of losses
            if self.log_dir and i % 10 == 0:
                log_writer.add_summary(stats)
            self.errors.append(loss)
            t2 = time()
            t_l = t2 - t1
            self._last_timings.append(t_l)
            t_lap = np.mean(self._last_timings)
            t_elapsed = t2 - t0
            t_total = t_lap * self.max_iter
            t_remain = t_total - t_elapsed
            if loss < self.tol:
                # Quit early if tolerance is met
                self._progressbar("stopping with loss < self.tol", i)
                break
            else:
                self._progressbar("loss: {}, time: {:.2f} s/itr, remain: {:.2f} hrs (elapsed: {:.2f} hrs out of total {:.2f} hrs)".format(                    
                    loss,
                    t_lap,
                    t_remain / 3600,
                    t_elapsed / 3600,
                    t_total / 3600), i)
            
            if (i % self.save_iters) == 0:
                self._save_status(i)
                if self.save_opt_hist:
                  self._save_optimization_history()
                    
                
        #endfor iters
        # Return the sum of the two learned matrices, as recommended
        # in the paper:
        self.save(self.name+'_embeds')
        return self._get_embeds()  
      
      
    def _get_embeds(self):
      return self.sess.run(tf.add(self.W, self.C))

    def _build_graph(self, vocab, initial_embedding_dict):
        """Builds the computatation graph.

        Parameters
        ------------
        vocab : Iterable
        initial_embedding_dict : dict
        """
        # Constants
        self.ones = tf.ones([self.n_words, 1])

        # Parameters:
        if initial_embedding_dict is None:
            # Ordinary GloVe
            self.W = self._weight_init(self.n_words, self.n, 'W')
            self.C = self._weight_init(self.n_words, self.n, 'C')
        else:
            # This is the case where we have values to use as a
            # "warm start":
            self.n = len(next(iter(initial_embedding_dict.values())))
            W = randmatrix(len(vocab), self.n)
            C = randmatrix(len(vocab), self.n)
            self.original_embedding = np.zeros((len(vocab), self.n))
            self.has_embedding = np.zeros(len(vocab))
            for i, w in enumerate(vocab):
                if w in initial_embedding_dict:
                    self.has_embedding[i] = 1.0
                    embedding = np.array(initial_embedding_dict[w])
                    self.original_embedding[i] = embedding
                    # Divide the original embedding into W and C,
                    # plus some noise to break the symmetry that would
                    # otherwise cause both gradient updates to be
                    # identical.
                    W[i] = 0.5 * embedding + noise(self.n)
                    C[i] = 0.5 * embedding + noise(self.n)
            self.W = tf.Variable(W, name='W', dtype=tf.float32)
            self.C = tf.Variable(C, name='C', dtype=tf.float32)
            self.original_embedding = tf.constant(self.original_embedding,
                                                  dtype=tf.float32)
            self.has_embedding = tf.constant(self.has_embedding,
                                             dtype=tf.float32)
            # This is for testing. It differs from
            # `self.original_embedding` only in that it includes the
            # random noise we added above to break the symmetry.
            self.G_start = W + C

        self.bw = self._weight_init(self.n_words, 1, 'bw')
        self.bc = self._weight_init(self.n_words, 1, 'bc')

        self.model = tf.tensordot(self.W, tf.transpose(self.C), axes=1) + \
            tf.tensordot(self.bw, tf.transpose(self.ones), axes=1) + \
            tf.tensordot(self.ones, tf.transpose(self.bc), axes=1)

    def _get_cost_function(self, weights, log_coincidence):
        """Compute the cost of the Mittens objective function.

        If self.mittens = 0, this is the same as the cost of GloVe.
        """
        self.weights = tf.Variable(weights,
                                   dtype=tf.float32,
                                   trainable=False)
        self.log_coincidence = tf.Variable(log_coincidence,
                                           dtype=tf.float32,
                                           trainable=False)
        self.diffs = tf.subtract(self.model, self.log_coincidence)
        cost = tf.reduce_sum(
            0.5 * tf.multiply(self.weights, tf.square(self.diffs)))
        if self.mittens > 0:
            self.mittens = tf.constant(self.mittens, tf.float32)
            cost += self.mittens * tf.reduce_sum(
                tf.multiply(
                    self.has_embedding,
                    self._tf_squared_euclidean(
                        tf.add(self.W, self.C),
                        self.original_embedding)))
        tf.summary.scalar("cost", cost)
        return cost

    def _get_cost_function_with_placeholders(self):
        """Compute the cost of the Mittens objective function.

        If self.mittens = 0, this is the same as the cost of GloVe.
        """
        self.weights = tf.placeholder(
            tf.float32, shape=[self.n_words, self.n_words])
        self.log_coincidence = tf.placeholder(
            tf.float32, shape=[self.n_words, self.n_words])
        
        self.diffs = tf.subtract(self.model, self.log_coincidence)
        cost = tf.reduce_sum(
            0.5 * tf.multiply(self.weights, tf.square(self.diffs)))
        if self.mittens > 0:
            self.mittens = tf.constant(self.mittens, tf.float32)
            cost += self.mittens * tf.reduce_sum(
                tf.multiply(
                    self.has_embedding,
                    self._tf_squared_euclidean(
                        tf.add(self.W, self.C),
                        self.original_embedding)))
        tf.summary.scalar("cost", cost)
        return cost


    @staticmethod
    def _tf_squared_euclidean(X, Y):
        """Squared Euclidean distance between the rows of `X` and `Y`.
        """
        return tf.reduce_sum(tf.pow(tf.subtract(X, Y), 2), axis=1)

    def _get_train_func(self):
        """Uses Adagrad to optimize the GloVe/Mittens objective,
        as specified in the GloVe paper.
        """
        optim = tf.train.AdagradOptimizer(self.learning_rate)
        if self.DEBUG:
            gradients = optim.compute_gradients(self.cost)
            if self.log_dir:
                for name, (g, v) in zip(['W', 'C', 'bw', 'bc'], gradients):
                    tf.summary.histogram("{}_grad".format(name), g)
                    tf.summary.histogram("{}_vals".format(name), v)
            return optim.apply_gradients(gradients)
        else:
            return optim.minimize(self.cost,
                                  global_step=tf.train.get_or_create_global_step())

    def _weight_init(self, m, n, name):
        """
        Uses the Xavier Glorot method for initializing weights. This is
        built in to TensorFlow as `tf.contrib.layers.xavier_initializer`,
        but it's nice to see all the details.
        """
        x = np.sqrt(6.0/(m+n))
        with tf.name_scope(name) as scope:
            return tf.Variable(
                tf.random_uniform(
                    [m, n], minval=-x, maxval=x), name=name)


class GloVe(Mittens, GloVeBase):

    __doc__ = GloVeBase.__doc__.format(
        framework=_FRAMEWORK,
        second=_DESC.format(model=GloVeBase._MODEL))



def _make_word_word_matrix(n=50):
    """Returns a symmetric matrix where the entries are drawn from a
    Poisson distribution"""
    base = np.random.zipf(2, size=(n, n)) - 1
    return base + base.T

if __name__ == '__main__':
    SIMPLE_TEST = False
    USE_FULL_GPU = True

    if SIMPLE_TEST:
        X = np.array([
            [10.0,  2.0,  3.0,  4.0],
            [ 2.0, 10.0,  4.0,  1.0],
            [ 3.0,  4.0, 10.0,  2.0],
            [ 4.0,  1.0,  2.0, 10.0]])
        embed_size = 4
    else:
        X = _make_word_word_matrix(13000)
        embed_size = 128
          
    glove = GloVe(n=embed_size, 
                  save_folder='mittens_models',
                  save_iters=100,
                  max_iter=1000, 
                  DEBUG=False, 
                  no_feeds=USE_FULL_GPU)
    G = glove.fit(X)
  
    print("\nLearned vectors:")
    print(G)
  
    print("We expect the dot product of learned vectors "
          "to be proportional to the co-occurrence counts. "
          "Let's see how close we came:")
  
    corr = np.corrcoef(G.dot(G.T).ravel(), X.ravel())[0][1]
    print("Pearson's R: {} ".format(corr))
  
