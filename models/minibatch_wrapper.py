import numpy as np
from foolbox.models import Model


class MinibatchWrapper(Model):
    """
    Helper that subdivides calls to batch_prediction into minibatches.
    """

    def __init__(self, model, batch_size):
        super(MinibatchWrapper, self).__init__(
            bounds=model.bounds(),
            channel_axis=model.channel_axis())

        self.wrapped_model = model
        self.batch_size = batch_size

    def __enter__(self):
        assert self.wrapped_model.__enter__() == self.wrapped_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped_model.__exit__(exc_type, exc_value, traceback)

    def num_classes(self):
        return self.wrapped_model.num_classes()

    def batch_predictions(self, images):

        m = images.shape[0]
        n_batches = (m // self.batch_size)
        if m % self.batch_size != 0:
            n_batches += 1

        batch_preds = np.empty((m, self.num_classes()), dtype=np.float32)
        for i_batch in range(n_batches):
            indices = slice(i_batch * self.batch_size, (i_batch+1) * self.batch_size)
            batch_preds[indices, :] = self.wrapped_model.batch_predictions(images[indices, ...])

        return batch_preds
