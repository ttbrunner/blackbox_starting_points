"""
Calculates saliency maps for all images in the ImageNet validation set.
This implementation creates a separate npz for each image, therefore multiple instances can be run in parallel (e.g. on a GPU cluster).
"""


import os
import numpy as np
import tensorflow as tf

from models.imagenet_irn_v2.foolbox_model import create_imagenet_irn_v2_model
from utils import dataset_imagenet, saliency


def main():
    # Load dataset
    n_classes = 1000
    img_shape = (299, 299, 3)
    imagenet_base_path = "/path/to/ILSVRC"
    y_val = dataset_imagenet.load_dataset_y_val(imagenet_base_path, limit=None)

    surr_model = create_imagenet_irn_v2_model()

    # Calculate normal saliency and guided backprop.
    tf_in = surr_model._images
    tf_logits = surr_model._batch_logits
    tf_sess = surr_model._session
    with tf_sess.graph.as_default():
        neuron_selector = tf.placeholder(tf.int32)
        tf_y_pred = tf_logits[0][neuron_selector]
    saliency_gb = saliency.GuidedBackprop(tf_sess.graph, tf_sess, tf_y_pred, tf_in)
    saliency_normal = saliency.GradientSaliency(tf_sess.graph, tf_sess, tf_y_pred, tf_in)

    out_path = "out_saliency_maps"
    os.makedirs(out_path, exist_ok=True)

    m = len(y_val)
    indices = np.arange(m)
    np.random.shuffle(indices)
    for i in indices:
        print("Processing img id {}.".format(i))
        sample_dir = os.path.join(out_path, "{}".format(i))
        try:
            os.makedirs(sample_dir, exist_ok=False)
        except OSError:
            continue

        X = dataset_imagenet.load_on_demand_X_val(imagenet_base_path, [i])[0]
        y_gt = y_val[i]

        # Calculate saliency maps against the ground truth class (no matter the actual prediction of the network).
        saliency_mask_normal = saliency_normal.GetMask(np.float32(X), feed_dict={neuron_selector: y_gt})
        saliency_mask_normal_smooth = saliency_normal.GetSmoothedMask(np.float32(X), feed_dict={neuron_selector: y_gt})
        saliency_mask_gb = saliency_gb.GetMask(np.float32(X), feed_dict={neuron_selector: y_gt})
        saliency_mask_gb_smooth = saliency_gb.GetSmoothedMask(np.float32(X), feed_dict={neuron_selector: y_gt})

        out_filepath = os.path.join(sample_dir, "maps.npz")
        np.savez_compressed(out_filepath,
                            saliency_mask_normal=saliency_mask_normal,
                            saliency_mask_normal_smooth=saliency_mask_normal_smooth,
                            saliency_mask_gb=saliency_mask_gb,
                            saliency_mask_gb_smooth=saliency_mask_gb_smooth)


if __name__ == '__main__':
    main()
