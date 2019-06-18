from collections import deque
import threading
import randomgen
import numpy as np

from utils.sampling.dct_utils import create_idct_noise


class SamplingProvider:
    """
    Multi-threaded util that runs in the background and precalculates noise samples. To be used in with statement!
    Even with the GIL, this is helpful because the GIL is released while we're waiting for model predictions.
    - If we're attacking a local model, TensorFlow releases GIL while blocking on sess.run
    - If we're attacking a remote model, blocking calls like HTTPRequest also release GIL
    - The RNG we use also releases GIL

    We can use that time to fill the queues with the next batch of samples.

    This implementation may be a bit over-engineered. The background is that we used to run this on a AWS instance that would sometimes
    crash if running >2 threads. For that reason, we wrote this util so we can fill (and block on) multiple deques with a single thread.
    """
    def __init__(self, shape, n_threads=1, queue_lengths=40):
        self.shape = shape
        self.n_threads = n_threads
        self.queue_lengths = queue_lengths
        self.queue_normal = deque()
        self.queue_lowfreq = deque()

        # Manually reimplemented Queue locking to cover 2 deques instead of 1.
        self.lock = threading.Lock()
        self.cv_not_full = threading.Condition(self.lock)           # Producer condition
        self.cv_not_empty = threading.Condition(self.lock)          # Consumer condition
        self.is_running = True
        self.threads = []
        for thread_id in range(n_threads):
            thread = threading.Thread(target=self._thread_fun, args=(thread_id,))
            thread.start()
            self.threads.append(thread)

    def __enter__(self):
        # Threads already started at __init__
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Stop all the threads.
        for thread in self.threads:
            thread.do_run = False

        self.is_running = False

        # Stop producers.
        with self.cv_not_full:
            self.cv_not_full.notify_all()

        # Stop consumers, if any.
        # Usually, Consumers are from the same thread that creates and __exit__s this object. So there will be none waiting at this point.
        # If, however, there is a consumer from a background thread, they could be still waiting, and then they will receive an
        #  InterruptedError here.
        with self.cv_not_empty:
            self.cv_not_empty.notify_all()

        for thread in self.threads:
            thread.join()
        print("SamplingProvider: all threads stopped.")

    def _thread_fun(self, thread_id):
        # create a thread-specifc RNG
        rng = randomgen.RandomGenerator(randomgen.Xoroshiro128(seed=20 + thread_id))
        rnd_normal = None
        rnd_lowfreq = None

        t = threading.currentThread()
        while getattr(t, 'do_run', True):

            # Prepare one of each sampling patterns
            if rnd_normal is None:
                rnd_normal = rng.standard_normal(size=self.shape, dtype='float32')
                rnd_normal /= np.linalg.norm(rnd_normal)
            if rnd_lowfreq is None:

                # MODIFIED
                # There are two concurring implementations of lowfreq noise, Perlin Noise by Brunner et al. and IDCT noise by Guo et al.
                # We found them to be pretty much equivalent, but Guo's has a much simpler implementation :)
                #
                # Therefore we replaced Perlin noise with Guo's lowfreq generator.
                # ALSO, the actual noise frequency is an important hyperparameter. Brunner et al. use a constant frequency for ImageNet.
                # We found that using a random (low-ish) frequency works much better, as we can span a much larger spectrum, and also avoid the
                # round "bubbles" found in the original "Guessing Smart" paper.
                freq = rng.uniform(-5., 1.)
                freq_ratio = 1. / (1. + np.exp(-freq))

                rnd_lowfreq = create_idct_noise(size_px=299, ratio=freq_ratio, normalize=True, rng_gen=rng)
                rnd_lowfreq = (rnd_lowfreq, freq)

            # Lock and put them into the queues.
            with self.cv_not_full:
                if len(self.queue_normal) >= self.queue_lengths and len(self.queue_lowfreq) >= self.queue_lengths:
                    self.cv_not_full.wait()

                # Fill one or both queues.
                if len(self.queue_normal) < self.queue_lengths:
                    self.queue_normal.append(rnd_normal)
                    rnd_normal = None
                if len(self.queue_lowfreq) < self.queue_lengths:
                    self.queue_lowfreq.append(rnd_lowfreq)
                    rnd_lowfreq = None

                self.cv_not_empty.notify_all()

    def get_normal(self, return_stats=False):
        """
        Returns a std-normal noise vector, normalized to L2=1
        """
        with self.cv_not_empty:
            while len(self.queue_normal) == 0:
                self.cv_not_empty.wait()
                if not self.is_running:
                    raise InterruptedError("Trying to consume an item, but was already shut down!")

            sample = self.queue_normal.popleft()
            self.cv_not_full.notify()

        # Unused in this implementation.
        # Use this field to record statistics of successful samples, see BlackboxWrapper.log_img
        if return_stats:
            return sample, {}
        else:
            return sample

    def get_lowfreq(self, return_stats=False):
        """
        Returns a lowfreq noise vector, normalized to L2=1
        """
        with self.cv_not_empty:
            while len(self.queue_lowfreq) == 0:
                self.cv_not_empty.wait()
                if not self.is_running:
                    raise InterruptedError("Trying to consume an item, but was already shut down!")

            item = self.queue_lowfreq.popleft()
            self.cv_not_full.notify()

        # Use this field to record statistics of successful samples, see BlackboxWrapper.log_img
        # Here, we remember the frequency of this sample. We could use this to automatically tune the frequency hyperparam (future work)
        sample = item[0]
        stats = {"freq_exp": float(item[1])}
        if return_stats:
            return sample, stats
        else:
            return sample
