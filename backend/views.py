from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from pathlib import Path
import dlib
from contextlib import contextmanager
from keras.utils.data_utils import get_file
from backend.wide_resnet import WideResNet
# from backend.addChannels import make3Channel
import tensorflow as tf
import difflib
import os
import json
import random
import collections
import time
from recordclass import recordclass

args = {'input_dir': 'bw_test', 'mode': 'test', 'output_dir': 'color', 'seed': random.randint(0, 2**31 - 1), 'checkpoint': 'train', 'max_steps': None, 'max_epochs': None, 'summary_freq': 100, 'progress_freq': 50, 'trace_freq': 0, 'display_freq': 0, 'save_freq': 5000, 'separable_conv': False, 'aspect_ratio': 1.0, 'lab_colorization': False, 'batch_size': 1, 'which_direction': 'AtoB', 'ngf': 64, 'ndf': 64, 'scale_size': 286, 'flip': True, 'lr': 0.0002, 'beta1': 0.5, 'l1_weight': 100.0, 'gan_weight': 1.0, 'output_filetype': 'png'}


EPS = 1e-12
CROP_SIZE = 256

A = recordclass('A',sorted(args))
a = A(**args)
Examples = collections.namedtuple("Examples", "inputs, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    # assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    # assertion = None
    # with tf.control_dependencies([assertion]):
    #     image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_examples(b64string):

    with tf.name_scope("load_images"):
        raw_input = tf.image.decode_image(b64string,channels=3)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        # assertion=None
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input)

    if a.which_direction == "AtoB":
        inputs = a_images
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    inputs_batch = tf.train.batch([ input_images], batch_size=a.batch_size)
    steps_per_epoch = 1

    return Examples(
        inputs=inputs_batch,
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(inputs.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, inputs)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, inputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(inputs - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []

    for i, _ in enumerate(fetches["outputs"]):
        name = str(i)
        fileset = {"name": name}
        kind="outputs"
        filename = name + "-" + kind + ".png"
        fileset[kind] = filename
        contents = fetches[kind][i]
        fileset['contents'] = contents
        filesets.append(fileset)
    return filesets

def mainColorize(b64string):
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    a[key]=val
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples(b64string)
    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.inputs)

    # undo colorization splitting on images that we use for display/output

    inputs = deprocess(examples.inputs)
    outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint",a.checkpoint)
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
            print("rate", (time.time() - start) / max_steps)
            dict={"img":str(base64.b64encode(filesets[0]['contents']).decode())}
            return dict

pretrained_model = "https://github.com/grinat/age_gender_detector/tree/master/models/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


import base64
import cv2
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np

def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        return cv2.resize(img, (int(w * r), int(h * r)))


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img




def main(b64string):
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                           file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    depth = 16
    k = 8
    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    image_dir = b64string
    margin = 0.4

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    image_generator = readb64(image_dir)

    img=image_generator
    input_img = img
    img_h, img_w, _ = np.shape(input_img)

    # detect faces using dlib detector
    detected = detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]),
                                    "M" if predicted_genders[i][0] < 0.5 else "F")
            draw_label(img, (d.left(), d.top()), label)

            string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
            dict = {
                'img': string
            }

            return dict

            # cv2.imshow("result", img)
            # key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
            #
            # if key == 27:  # ESC
            #     break

@csrf_exempt
def guess_age(request):
    """
    List all code snippets, or create a new snippet.
    """
    tf.reset_default_graph()
    tf.Graph().as_default()
    if request.method == 'GET':
        guess = {}
        return JsonResponse(guess, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        base64Str=data['img']
        return JsonResponse(main(base64Str), status=201)


@csrf_exempt
def colorize(request):
    """
    List all code snippets, or create a new snippet.
    """
    tf.reset_default_graph()
    tf.Graph().as_default()

    if request.method == 'POST':
        data = JSONParser().parse(request)
        # bs64="data:image/jpeg;base64,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"
        bs64 = data['img'].split(',')[1]
        return JsonResponse(mainColorize(base64.decodestring(bytes(bs64, 'ascii'))), status=201)


def home(request):
    return render(request, 'backend/home.html', {})