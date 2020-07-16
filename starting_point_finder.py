import gc
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.image import translate as tf_translate
from scipy.ndimage import gaussian_filter

from utils.distance_measures import DistL2
from models.minibatch_wrapper import MinibatchWrapper
from utils import saliency, dataset_imagenet
from utils.util import find_img_centroid


class StartingPointFinder:
    def __init__(self, bb_model, surr_model, imagenet_base_path, saliency_base_path, tf_session, batch_size=8):
        self.bb_model = MinibatchWrapper(bb_model, batch_size=batch_size)
        self.surr_model = MinibatchWrapper(surr_model, batch_size=batch_size)
        self.imagenet_base_path = imagenet_base_path
        self.saliency_base_path = saliency_base_path

        # TF graph for fast-ish random transforms
        self.tf_session = tf_session
        self.tf_img = None
        self.tf_centroid = None
        self.tf_resize = None
        self.tf_flip = None
        self.tf_centr_target = None
        self.tf_trans_out = None
        self.tf_shifted_centr_out = None

        self.init_transform_graph()

    def init_transform_graph(self):
        """
        TF graph for random scale, flip, translate for a batch of images.
        """
        img_shape = (299, 299, 3)

        with self.tf_session.as_default():
            tf_img = tf.placeholder(dtype=tf.float32, shape=(None,) + img_shape)
            tf_centroid = tf.placeholder(dtype=tf.float32, shape=2)                         # Pixel coords of centroid

            tf_resize = tf.placeholder(dtype=tf.int32, shape=2)                             # resize: new_height, new_width
            tf_flip = tf.placeholder(dtype=tf.bool, shape=())                               # flip horizontally: true/false
            tf_centr_target = tf.placeholder(dtype=tf.int32, shape=2)                       # translate img so centroid is at this location: y/x

            flipped = tf.cond(tf_flip, lambda: tf.image.flip_left_right(tf_img), lambda: tf_img)
            flipped_centr = tf_centroid * [1., 0.]
            flipped_centr += tf.cond(tf_flip, lambda: (img_shape[:2] - tf_centroid), lambda: tf_centroid) * [0., 1.]

            scaled = tf.image.resize_images(flipped, tf_resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            scaled_centr = flipped_centr * (tf.cast(tf_resize, tf.float32) / img_shape[:2])

            padded = tf.image.pad_to_bounding_box(scaled, offset_height=0, offset_width=0,
                                                  target_height=img_shape[0], target_width=img_shape[1])
            padded_centr = scaled_centr

            to_shift = tf.cast(tf_centr_target, tf.float32) - padded_centr
            shifted_centr = padded_centr + to_shift
            to_shift = tf.ones([tf.shape(padded)[0], 2]) * to_shift                         # Tile into batch dimension
            shifted_img = tf_translate(padded, to_shift[:, ::-1], interpolation="BILINEAR") # For some weird reason this is xy and not yx

            self.tf_img = tf_img
            self.tf_centroid = tf_centroid
            self.tf_resize = tf_resize
            self.tf_flip = tf_flip
            self.tf_centr_target = tf_centr_target

            self.tf_trans_out = shifted_img
            self.tf_shifted_centr_out = shifted_centr

    def random_transform(self, imgs, centroid_start, centroid_orig):
        """
        Randomly transforms a batch of images.
        An extra translation is added (centroid_start - centroid_orig), in an effort to try and align the saliency centroids
        of both starting point and original image. The basic idea was to have features of the adversarial class directly on top
        of features of the original, hopefully overwriting them with minimal perturbation required.

        :param imgs: A batch of images (b,h,w,c)
        :param centroid_start: Saliency centroid of the image(s). Only one centroid for the entire batch.
        :param centroid_orig: Saliency centroid of the image to attack.
        :return: transformed batch of images.
        """

        rnd_resize = np.random.uniform(0.5, 1.0)
        rnd_resize = np.int32(np.round(rnd_resize * 299))
        rnd_flip = np.random.uniform() < 0.5
        rnd_shift = [np.random.randint(-75, 75), np.random.randint(-75, 75)]

        with self.tf_session.as_default():
            imgs_trans, centroid_target_trans = self.tf_session.run([self.tf_trans_out, self.tf_shifted_centr_out], feed_dict={
                self.tf_img: imgs,
                self.tf_centroid: centroid_start,
                self.tf_resize: [rnd_resize, rnd_resize],               # Keep aspect ratio
                self.tf_flip: rnd_flip,
                self.tf_centr_target: centroid_orig + rnd_shift
            })
        centroid_target_trans = np.int32(np.round(centroid_target_trans))
        return imgs_trans, centroid_target_trans

    def load_saliency_mask(self, img_id):
        """ Loads a precalc'd saliency map from disk. Need to run precalc_saliency_maps first. """

        filepath = os.path.join(self.saliency_base_path, str(img_id), "maps.npz")
        maps = np.load(filepath)

        # Use SmoothGrad normal and SmoothGrad Guided Backprop
        saliency_mask_normal = saliency.VisualizeImageGrayscale(maps["saliency_mask_normal_smooth"])
        saliency_mask_gb = saliency.VisualizeImageGrayscale(maps["saliency_mask_gb_smooth"])

        # Combine normal and GB masks. SmoothGrad GB has really good features, but often large areas in the background that don't matter.
        # Normal smooth doesn't have those, so we try to keep GB features, but remove background areas.
        saliency_mask_combined = np.copy(saliency_mask_gb)
        low_index = saliency_mask_normal < 0.05
        saliency_mask_combined[low_index] = saliency_mask_normal[low_index]

        return saliency_mask_combined

    def find_start_from_saliency(self, X_orig, orig_id, y_target, target_ids):
        """
        Creates a starting point for attacking an image using precalculated saliency information.
        :param X_orig: The original image to attack.
        :param orig_id: The image ID (in the X_val dataset) of the original. Needed for retrieving saliency maps.
        :param y_target: The target adversarial class label.
        :param target_ids: Image IDs (in the X_val dataset) of all images of the target class.
        :return:
        """

        if len(target_ids) == 0:
            raise ValueError("No images of the target class!")
        print("There are {} images of the target class.".format(len(target_ids)))
        if len(target_ids) > 500:
            print("WARN: Large number of images of the target class! Do you have enough memory?")

        dm_l2 = DistL2().to_range_255()

        print("Loading precalc'd saliency maps...")
        X_target_all = dataset_imagenet.load_on_demand_X_val(self.imagenet_base_path, target_ids)
        saliency_target_all = np.empty(X_target_all.shape[:3], dtype=np.float32)
        centroid_target_all = np.empty((X_target_all.shape[0], 2), dtype=np.int32)

        # We run this function in a loop 5 times. Sometimes, but not often, we may randomly fail to construct a good starting point.
        # When this happens, we amplify and smoothen the saliency mask even further, resulting in a larger patch that is added to the image.
        # If we still fail 4 times, in the last try we set the mask to 100%. This is basically a fallback to a "closest image" strategy.
        found_start = False
        n_tries = 5
        for i_try in range(n_tries):

            for i, img_id in enumerate(target_ids):
                saliency_mask_gray = self.load_saliency_mask(img_id)
                centroid_target_all[i, :] = find_img_centroid(saliency_mask_gray, min_mass_threshold=.5)     # Save center of mass, can be [-1,-1] if empty

                # Amplify mask:
                # - Default: Amplify via sqrt and add smooth transitions via Gauss
                # - If previous tries didn't find anything: amplify even further.
                # - If last try: set mask to full img.
                if i_try >= 1:
                    saliency_mask_gray = saliency_mask_gray ** 0.75
                if i_try >= 2:
                    saliency_mask_gray = saliency_mask_gray ** 0.75
                if i_try >= 3:
                    saliency_mask_gray += 0.1 * i_try
                if i_try == n_tries - 1:
                    saliency_mask_gray = np.ones_like(saliency_mask_gray)

                saliency_mask_gray = saliency_mask_gray ** 0.55
                saliency_mask_gray += 2. * gaussian_filter(saliency_mask_gray, sigma=1.0)

                np.clip(saliency_mask_gray, 0., 1., out=saliency_mask_gray)
                saliency_target_all[i, ...] = saliency_mask_gray

            saliency_orig = self.load_saliency_mask(orig_id)
            saliency_orig_color = np.tile(saliency_orig[:, :, np.newaxis], (1, 1, 3))
            centroid_orig = find_img_centroid(saliency_orig, min_mass_threshold=.5)
            print("Original image's saliency centroid is at {}.".format(centroid_orig))

            X_orig_mod = np.copy(np.float32(X_orig))

            # Random seed: fix so the generated starting points are the same each time we run this.
            #  HOWEVER, change the (fixed) seed for every subsequent try, so we don't retry the same transforms that didn't work before.
            np.random.seed(i_try)

            # Generate all candidates
            X_start_all = []
            masks_all = []
            img_ids_all = []
            for i, img_id in enumerate(target_ids):

                print("Preparing target img {}...".format(img_id))
                saliency_mask_color = np.tile(saliency_target_all[i, :, :, np.newaxis], (1, 1, 3))
                X_target = X_target_all[i, ...]
                centroid_start = centroid_target_all[i, :]
                if centroid_start[0] < 0:
                    print("WARN: Saliency mask is empty for this image. Skipping.")
                    continue

                print("Target image's saliency centroid is at {}.".format(centroid_start))

                # Do a lot of random transforms (zoom and rotate).
                n_trans_samples = 50
                for i_trans_sample in range(n_trans_samples):

                    # Do a random transform (scale, flip and shift)
                    # - For an L2 attack, we want to create one small patch with strong features of the target.
                    #       This massively reduces the L2 distance of the starting point.
                    X_start = np.copy(X_orig_mod)
                    mask_sum = np.zeros_like(X_start)
                    n_copies = 1                                # If you want, you can also add multiple copies :) Should work well for L_inf attacks.
                    for i_copy in range(n_copies):
                        imgs = np.stack([X_target, saliency_mask_color])
                        imgs_trans, centroid_target_trans = self.random_transform(imgs, centroid_start=centroid_start, centroid_orig=centroid_orig)
                        X_target_trans, saliency_map_trans = imgs_trans

                        # Interpolate: Starting point = original + saliency*(target_img - original)
                        diff = (np.float32(X_target_trans) - np.float32(X_start)) * saliency_map_trans
                        X_start = np.clip(X_start + diff, 0., 255.)
                        mask_sum = np.clip(mask_sum + saliency_map_trans, 0., 1.)
                    X_start_all.append(X_start)
                    masks_all.append(mask_sum)
                    img_ids_all.append(img_id)                      # Remember the image id for logging purposes

            X_start_all = np.uint8(np.clip(np.round(np.array(X_start_all)), 0, 255))
            masks_all = np.array(masks_all)
            img_ids_all = np.array(img_ids_all)

            # Get predictions from surrogate model and remove all starting points that are not adversarial on it.
            # This is a quick sanity check, so we don't waste too many calls on the black box model.
            preds_all = self.surr_model.batch_predictions(X_start_all)
            adv_filter = np.argmax(preds_all, axis=1) == y_target
            print("{} of {} starting points are adversarial for the surrogate.".format(np.sum(adv_filter), len(X_start_all)))
            X_start_all = X_start_all[adv_filter]
            masks_all = masks_all[adv_filter]
            img_ids_all = img_ids_all[adv_filter]

            # Calculate distance for remaining images. Lower is better.
            score_all = np.empty(X_start_all.shape[0])
            for i in range(X_start_all.shape[0]):
                score_all[i] = dm_l2.calc(X_start_all[i, ...], X_orig)

            # Sort ascending by distance, and try them one by one on the black box. The first to succeed is picked.
            sort_indices = np.argsort(score_all)
            for i, ind in enumerate(sort_indices):
                X_best = X_start_all[ind]
                mask_best = masks_all[ind]
                img_id_best = img_ids_all[ind]
                pred = self.bb_model.predictions(X_best)
                if np.argmax(pred) == y_target:
                    found_start = True
                    print("Found a valid starting point (no. {}).".format(i))
                    break

            if found_start:
                break
            else:
                print("None of the starting points were adversarial on the black box model.")
                print("Reducing mask strength, retry no. {}".format(i_try+1))
                gc.collect()

        if not found_start:
            # This should only happen if not even the clean images worked.
            raise ValueError("Could not find a starting point - at all.")

        # Mask: features of target class AND original features (allow to perturb both). Also scale it between [0,1] for the BBA.
        mask_best = np.maximum(mask_best, saliency_orig_color)
        mask_best /= np.max(mask_best)

        dist = dm_l2.calc(X_orig, X_best)
        print("Picked img with d_l2 of {:.3f}. Used features from image id {}.".format(dist, img_id_best))
        print("Dimensionality of search space: {}".format(np.sum(mask_best)))

        return np.uint8(np.clip(np.round(X_best), 0., 255.)), mask_best, img_id_best
