from keras.models import *
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Dropout, \
    Concatenate, BatchNormalization, Lambda, Add, Activation
from keras import layers
from keras.optimizers import *
import tensorflow as tf
from functools import partial
# from group_normalization import GroupNormalization
# from niftynet.network.vnet import VNet
# from niftynet.network.highres3dnet import HighRes3DNet
from keras import Input
import numpy as np
from keras import backend as K


def group_norm(x, groups=8, eps=1e-5, gamma=None, beta=None):
    """
    https://arxiv.org/pdf/1803.08494.pdf
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1] #
    G: number of groups for GN

    :param x:
    :param gamma:
    :param beta:
    :param G:
    :param eps:
    :return:
    """
    N, H, W, D, C = x.get_shape().as_list()
    x = tf.reshape(x, [-1, H, W, D, groups, C // groups])
    mean_res, var_res = tf.nn.moments(x, [1, 2, 3, 5], keep_dims=True)
    x = (x - mean_res) / tf.sqrt(var_res + eps)
    x = tf.reshape(x, [-1, H, W, D, C])
    return x


def weighted_dice(labels, prediction):
    eps = tf.keras.backend.epsilon()
    labels = tf.cast(labels, tf.float32)
    prediction = tf.cast(prediction, tf.float32)

    # Normalized region-wise squared weights [1/(region_l_vol)**2]/
    weights = tf.reduce_sum(labels, axis=[1,2,3])
    sqrd_weights = tf.square(weights)
    weights = tf.reciprocal(sqrd_weights)
    weights = weights/(tf.expand_dims(tf.reduce_sum(weights, -1), -1))
    weights = tf.ones_like(weights)

    # To avoid numerical instability
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)

    # Region-wise weighted intersection
    numerator = tf.reduce_sum(labels * prediction, axis=[1, 2, 3])
    numerator = tf.multiply(weights, numerator)
    # numerator = tf.reduce_sum(numerator, axis=-1)
    numerator = 2 * numerator

    # Region-wise weighted Union
    denominator = tf.reduce_sum(labels + prediction, axis=[1, 2, 3])
    denominator = tf.multiply(weights, denominator)
    # denominator = tf.reduce_sum(denominator, axis=-1)

    dc = numerator/(denominator + eps)
    dc = tf.reduce_mean(dc, axis=-1)

    return dc


def weighted_dice_loss(labels, prediction):
    return 1.0 - weighted_dice(labels, prediction)


def compute_weighted_dice_loss(labels, prediction):
    return tf.reduce_mean(weighted_dice_loss(labels, prediction))


def signed_dist_function(f):
    from scipy import ndimage
    """Return the signed distance to the 0.5 levelset of a function."""
    # Prepare the embedding function.
    f = f > 0.5
    f = f.astype(np.float32)

    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f).astype(np.float32) - 0.5, -(dist_func(
        1-f).astype(np.float32) - 0.5))
    distance = distance.astype(np.float32)
    return -distance


def tf_signed_dist_function(f):
    """ Signed distance function implemented in tensorflow """
    from scipy import ndimage
    """Return the signed distance to the 0.5 levelset of a function."""
    # Prepare the embedding function.
    f = f > 0.5
    f = f.astype(np.float32)

    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f).astype(np.float32) - 0.5, -(dist_func(
        1-f).astype(np.float32) - 0.5))
    distance = distance.astype(np.float32)
    return -distance


def sdf(labels, prediction):
    labels = tf.cast(labels, tf.float32)
    prediction = tf.cast(prediction, tf.float32)

    # # Dice loss
    # # dc = dice_loss(labels, prediction)

    num_channels = K.int_shape(prediction)[-1]
    # Signed distance function of every region
    arrs = []
    for channel_index in range(num_channels):
        arrs.append(tf.py_func(signed_dist_function, [labels[...,
                                                             channel_index]],
                               tf.float32))
    sdf_tensor = tf.stack(arrs, axis=-1)
    # relu_res_sdf = tf.nn.relu(sdf_tensor)

    # Take relu of SDF so error outside surface region will only be considered.
    sdf_loss = labels * prediction

    # Sum of sdf loss per region
    sdf_loss = tf.reduce_sum(sdf_loss, axis=[1, 2, 3, 4])

    return sdf_loss/tf.reduce_max(sdf_loss)

def compute_sdf_loss(label, prediction):
    """ SDF-based loss and dice loss"""
    return tf.reduce_mean(sdf(label, prediction))


# Loss function when SDF_LOSS is set to False.
# SDF is calculated inside the loss function in this function during training
def composite_sdf_dice(labels, prediction):
    labels = tf.cast(labels, tf.float32)
    prediction = tf.cast(prediction, tf.float32)

    # # Dice loss
    num_channels = K.int_shape(prediction)[-1]

    # Remove background loss during training
    dc_loss = dice_loss(labels, prediction)

    # Signed distance function of every region
    arrs = []
    for channel_index in range(num_channels):
        arrs.append(tf.py_func(signed_dist_function, [labels[...,
                                                             channel_index]],
                               tf.float32))
    sdf_tensor = tf.stack(arrs, axis=-1)
    # relu_res_sdf = tf.nn.relu(sdf_tensor)

    # Take relu of SDF so error outside surface region will only be considered.
    sdf_loss = tf.nn.relu(sdf_tensor * prediction)
    # sdf_loss = sdf_loss/tf.reduce_max(sdf_loss) # normalize to
    # # tensormaximum

    # Sum of sdf loss per region
    sdf_loss = tf.reduce_sum(sdf_loss, axis=[1, 2, 3, 4])
    sdf_loss = sdf_loss/tf.reduce_max(sdf_loss) # normalize loss
    return sdf_loss + dc_loss


def compute_composite_loss(label, prediction):
    """ SDF-based loss and dice loss"""
    return tf.reduce_mean(composite_sdf_dice(label, prediction))


def dice(labels, prediction):
    eps = tf.keras.backend.epsilon()
    labels = tf.cast(labels, tf.float32)
    prediction = tf.cast(prediction, tf.float32)

    # Exclude background region when calculating loss term
    # labels = labels[..., 1:]
    # prediction = prediction[..., 1:]

    # Region-wise weighted intersection
    numerator = tf.reduce_sum(labels * prediction, axis=[1, 2, 3])
    numerator = 2 * numerator

    # Region-wise weighted Union
    denominator = tf.reduce_sum(labels + prediction, axis=[1, 2, 3])

    dc = (numerator + eps) / (denominator + eps)
    dc = tf.reduce_mean(dc, axis=-1)
    return dc


def dice_loss(labels, prediction):
    return 1.0 - dice(labels, prediction)


def compute_dice_loss(labels, prediction):
    return tf.reduce_mean(dice_loss(labels, prediction))


def dice_w_softmax(labels, prediction):
    eps = tf.keras.backend.epsilon()
    labels = tf.cast(labels, tf.float32)
    prediction = tf.cast(prediction, tf.float32)

    # Softmax output
    # labels = labels[..., 1:]
    prediction = tf.nn.softmax(prediction)

    # Region-wise weighted intersection
    numerator = tf.reduce_sum(labels * prediction, axis=[1, 2, 3])
    numerator = 2 * numerator

    # Region-wise weighted Union
    denominator = tf.reduce_sum(labels + prediction, axis=[1, 2, 3])

    dc = (numerator + eps) / (denominator + eps)
    dc = tf.reduce_mean(dc, axis=-1)
    return dc

def dice_w_softmax_loss(labels, prediction):
    return 1.0 - dice_w_softmax(labels, prediction)


def compute_dice_w_softmax_loss(labels, prediction):
    return tf.reduce_mean(dice_w_softmax_loss(labels, prediction))


def dice_callback(labels, prediction):
    """ Function to use to plot dice coefficients during training in keras. """
    eps = tf.keras.backend.epsilon()
    labels = tf.cast(labels, tf.float32)
    prediction = tf.cast(prediction, tf.float32)

    # Region-wise weighted intersection
    numerator = tf.reduce_sum(labels * prediction, axis=[1, 2, 3])
    numerator = 2 * numerator

    # Region-wise weighted Union
    denominator = tf.reduce_sum(labels + prediction, axis=[1, 2, 3])

    dc = (numerator + eps)/(denominator + eps)
    dc = tf.reduce_mean(dc, axis=-1)
    return dc


# Implementation of exponential logarithmic loss
# (https://arxiv.org/pdf/1809.00076.pdf)
def exponential_log_dice(labels, prediction):
    eps = tf.keras.backend.epsilon()
    labels = tf.cast(labels, tf.float32)
    prediction = tf.cast(prediction, tf.float32)
    # prediction = tf.nn.softmax(prediction)

    # Region-wise weighted intersection
    numerator = tf.reduce_sum(labels * prediction, axis=[1, 2, 3])
    numerator = 2 * numerator

    # Region-wise weighted union
    denominator = tf.reduce_sum(labels + prediction, axis=[1, 2, 3])

    # Exponential logarithmic dice using gamma 0.3
    dc = -tf.log((numerator + 1)/(denominator + 1))**0.3

    # exp log dice of current all mini-batch
    dc = tf.reduce_mean(dc, axis=-1)

    # dc size is (batch_size)
    return dc


def wt_exp_cross_entropy(labels, prediction):
    """ Weighted exponential cross-entropy. """
    # exp_cross_entropy = tf.keras.backend.categorical_crossentropy(
    #     target=labels, output=prediction)**0.3
    exp_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=prediction) ** 0.3

    # Weighting per label (batch_size, n_channels)
    label_weights = tf.reduce_sum(labels, axis=[1, 2, 3])
    label_weights_total = tf.reduce_sum(label_weights, axis=-1)
    label_weights_total = tf.maximum(label_weights_total, 1e-6)
    label_weights = label_weights/label_weights_total
    label_weights = label_weights ** 0.5

    # Label weights of every voxel. Shape is (batch_size, h, w, d)
    label_weights = tf.reduce_sum(label_weights * labels, axis=-1)

    # Mean cross entropy of current batch
    exp_cross_entropy = tf.reduce_mean(tf.multiply(label_weights,
                                                   exp_cross_entropy),
                                       axis=[1, 2, 3])

    # exp_cross_entropy size is (batch_size)
    return exp_cross_entropy


def compute_exponential_log_loss(labels, prediction):
    # Mean exponential logarithmic loss
    exp_dice_loss = exponential_log_dice(labels, prediction)
    # Weighted exponential cross-entropy
    # wt_exp_ce = wt_exp_cross_entropy(labels, prediction)

    # Weighted losses
    # loss = (0.8 * exp_dice_loss) + (0.2 * wt_exp_ce)
    loss = 0.8 * exp_dice_loss
    loss = tf.reduce_mean(loss)
    return loss


M = np.ones((23, 23), dtype=np.float32)
M = M - np.eye(23, dtype=np.float32)

def label_wise_dice_coefficient(y_true, y_pred, label_index):
    # y_true = tf.nn.softmax(y_true)
    # y_pred = tf.nn.softmax(y_pred)
    return dice_callback(y_true[..., label_index:label_index + 1], y_pred[...,
                                                                   label_index:label_index + 1])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for
                           index in range(23)]


def softmax_label_wise_dice_coefficient(y_true, y_pred, label_index):
    """ Only use if network output is a multi-label approach. """
    y_pred = tf.nn.softmax(y_pred)
    return dice_callback(y_true[..., label_index:label_index+1], y_pred[...,
                                                                 label_index:label_index+1])

def get_softlabel_dice_coefficient_function(label_index):
    f = partial(softmax_label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'softmax_label_{0}_dice_coef'.format(label_index))
    return f

label_wise_softmax_dice_metrics = [get_softlabel_dice_coefficient_function(index)
                                   for index in range(23)]


def label_wise_composite_loss(y_true, y_pred, label_index):
    true_label = y_true[..., label_index:label_index+1]
    pred_label = y_pred[..., label_index:label_index+1]
    return composite_sdf_dice(true_label, pred_label)


def get_label_composite_loss_function(label_index):
    f = partial(label_wise_composite_loss, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_composite_loss'.format(label_index))
    return f


label_wise_composite_loss_metrics = [get_label_composite_loss_function(
    index) for index in range(19)]


def build_vnet_network(inputs=Input((32, 32, 32, 1)), depth=2,
                       filters=[64, 128, 256], dropoutAt=[None, 0.5, None],
                       multidice=True, n_class=23, batch_normalize=False,
                       group_normalize=False, l_rate=1e-3, n_groups=8):
    steps_down = []
    steps_up = []
    outputs = []
    # check sanity of parameters:
    if not depth + 1 == len(filters):
        raise Exception("Length of filters array does not match to depth.")
    if not depth + 1 == len(dropoutAt):
        raise Exception("Length of dropoutAt array does not match to depth.")

    # STEP DOWN THE V NET NETWORK
    # preparing the loop variable to start with the inputs
    recentlayer_PREV_i_down = inputs
    for i in range(depth):
        n_filters = filters[i]
        conv_i_down = Conv3D(n_filters, 5, activation='relu', padding='same',
                             kernel_initializer='he_normal')(
            recentlayer_PREV_i_down)
        if batch_normalize:
           conv_i_down = BatchNormalization()(conv_i_down)

        if group_normalize:
            conv_i_down = Lambda(group_norm)(conv_i_down)
        conv_i_down = Conv3D(n_filters, 5, activation='relu', padding='same',
                             kernel_initializer='he_normal')(conv_i_down)
        if group_normalize:
            conv_i_down = Lambda(group_norm)(conv_i_down)
        # Residuals - broadasting only works for 1 input channel
        recentlayer_PREV_i_down = Conv3D(n_filters, 1, activation='relu',
                                         padding='same',
                                         kernel_initializer='he_normal')(
            recentlayer_PREV_i_down)
        conv_i_down = Add()([conv_i_down, recentlayer_PREV_i_down])

        # perform dropout if desired
        if dropoutAt[i] is not None:
            # perform dropout
            drop_i_down = Dropout(dropoutAt[i])(conv_i_down)
            pool_i_down = Conv3D(n_filters, kernel_size=(2, 2, 2), strides=(2,
                                                                            2,
                                                                            2),
                                 kernel_initializer='he_normal')(drop_i_down)
            # pool_i_down = MaxPooling3D(pool_size=(2, 2, 2))(drop_i_down)

        else:
            # skip dropout
            pool_i_down = Conv3D(n_filters, kernel_size=(2, 2, 2), strides=(2,
                                                                            2,
                                                                            2),
                                 kernel_initializer='he_normal')(conv_i_down)

            if group_normalize:
                pool_i_down = Lambda(group_norm)(pool_i_down)

            # pool_i_down = MaxPooling3D(pool_size=(2, 2, 2))(conv_i_down)
        # save conv layer for later feature concatenation
        steps_down.append(conv_i_down)

        # propagate recent layer to next iteration
        recentlayer_PREV_i_down = pool_i_down

    # We are at the bottom of the V net
    # -> perform last convolution
    conv_deepest = Conv3D(filters[depth], 5, activation='relu', padding='same',
                          kernel_initializer='he_normal')(
        recentlayer_PREV_i_down)
    if batch_normalize:
        conv_deepest = BatchNormalization()(conv_deepest)

    if group_normalize:
        conv_deepest = Lambda(group_norm)(conv_deepest)

    conv_deepest = Conv3D(filters[depth], 5, activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv_deepest)

    recentlayer_PREV_i_down = Conv3D(filters[depth], 1, activation='relu',
                                     padding='same',
                          kernel_initializer='he_normal')(
        recentlayer_PREV_i_down)
    recentlayer_PREV_i_down = Lambda(group_norm)(recentlayer_PREV_i_down)
    conv_deepest = Add()([conv_deepest, recentlayer_PREV_i_down])

    # appyling dropout and
    # preparing the loop variable to start with the deepest dropout layer
    if dropoutAt[depth] is not None:
        # perform dropout
        drop_deepest = Dropout(dropoutAt[depth])(conv_deepest)
        recentlayer_PREV_i_up = drop_deepest
    else:
        # skip dropout
        recentlayer_PREV_i_up = conv_deepest

    # STEP UP THE V NETWORK
    for i in reversed(range(depth)):
        n_filters = filters[i]
        corresponding_conv_i_down = steps_down[i]

        up_i_up = Conv3D(n_filters, 1, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
            UpSampling3D(size=(2, 2, 2))(recentlayer_PREV_i_up))

        if group_normalize:
            up_i_up = Lambda(group_norm)(up_i_up)

        # merge_i_up = merge([corresponding_conv_i_down, up_i_up], mode='concat', concat_axis=4)
        # merge_i_up = Concatenate(axis=4)([corresponding_conv_i_down, up_i_up])
        merge_i_up = Add()([corresponding_conv_i_down, up_i_up])
        conv_i_up = Conv3D(n_filters, 5, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge_i_up)

        if batch_normalize:
            conv_i_up = BatchNormalization()(conv_i_up)

        if group_normalize:
            conv_i_up = Lambda(group_norm)(conv_i_up)

        conv_i_up = Conv3D(n_filters, 5, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv_i_up)
        conv_i_up = Lambda(group_norm)(conv_i_up)

        merge_i_up = Conv3D(n_filters, 1, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge_i_up)

        conv_i_up = Add()([conv_i_up, merge_i_up])

        # save conv layer (but not needed)
        steps_up.append(conv_i_up)

        # propagate recent layer to next iteration
        recentlayer_PREV_i_up = conv_i_up

    if multidice:
        # for multi dice output
        for i, conv_up_layer in enumerate(steps_up):
            # we may need to upscale for some outputs
            resizeFactor = 2 ** (len(steps_up) - (i + 1))

            output_layer_i = Conv3D(2, 3, activation='relu', padding='same',
                                    kernel_initializer='he_normal')(
                conv_up_layer)

            # add upsampling layer, if needed and add name to output layer
            if resizeFactor > 1:
                output_layer_i = Conv3D(1, 1, activation='sigmoid')(
                    output_layer_i)
                output_layer_i = UpSampling3D(size=resizeFactor,
                                              name="d{}_aux_out".format(
                                                  int(resizeFactor / 2)))(
                    output_layer_i)
            else:
                output_layer_i = Conv3D(1, 1, activation='sigmoid',
                                        name='d0_out')(output_layer_i)

            outputs.append(output_layer_i)
    else:
        # simple output -> compute output for each layer
        conv_last = Conv3D(n_filters, 5, activation='relu', padding='same',
                           kernel_initializer='he_normal')(
            recentlayer_PREV_i_up)
        # if batch_normalize:
        #     conv_last = BatchNormalization()(conv_last)
        if group_normalize:
            conv_last = Lambda(group_norm)(conv_last)

        recentlayer_PREV_i_up = Conv3D(n_filters, 1, activation='relu',
                                       padding='same',
                            kernel_initializer='he_normal')(
            recentlayer_PREV_i_up)

        conv_last = Add()([conv_last, recentlayer_PREV_i_up])

        if group_normalize:
            conv_last = Lambda(group_norm)(conv_last)

        conv_last = Conv3D(n_class, 1, activation='softmax')(conv_last)
        outputs = [conv_last]

    model = Model(inputs=inputs, outputs=outputs)

    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss=compute_dice_loss,
    #               metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=l_rate),
                  loss=compute_dice_loss,
                  metrics=label_wise_dice_metrics +
                          label_wise_softmax_dice_metrics + [
                              compute_dice_w_softmax_loss])
    return model

