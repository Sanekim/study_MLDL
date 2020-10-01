from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
import tensorflow as tf
from model.common import normalize, denormalize, subpixel_conv2d
# from common import normalize, denormalize, subpixel_conv2d              # 19.09.04. 수정하면 오류 발생


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    # pre ResBlock Concat 1 ########################################
    a = Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    a = Conv2D(num_filters, 3, padding='same')(a)
    a = Add()([a, x])

    b = Conv2D(num_filters, 3, padding='same', activation='relu')(a)
    b = Conv2D(num_filters, 3, padding='same')(b)
    b = Add()([b, a])

    c = Conv2D(num_filters, 3, padding='same', activation='relu')(b)
    c = Conv2D(num_filters, 3, padding='same')(c)
    c = Add()([c, b])

    d = Conv2D(num_filters, 3, padding='same', activation='relu')(c)
    d = Conv2D(num_filters, 3, padding='same')(d)
    d = x = Add()([d, c])

    conc = tf.concat([b, a], -1)
    conc = tf.concat([c, conc], -1)
    x_conc = tf.concat([d, conc], -1)

    # Upsample 1st
    x = Lambda(subpixel_conv2d(scale=2))(x_conc)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(6):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    # post ResBlock Concat 1
    a = Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    a = Conv2D(num_filters, 3, padding='same')(a)
    a = Add()([a, x])

    b = Conv2D(num_filters, 3, padding='same', activation='relu')(a)
    b = Conv2D(num_filters, 3, padding='same')(b)
    b = Add()([b, a])

    c = Conv2D(num_filters, 3, padding='same', activation='relu')(b)
    c = Conv2D(num_filters, 3, padding='same')(c)
    c = Add()([c, b])

    d = Conv2D(num_filters, 3, padding='same', activation='relu')(c)
    d = Conv2D(num_filters, 3, padding='same')(d)
    d = Add()([d, c])

    conc = tf.concat([b, a], -1)
    conc = tf.concat([c, conc], -1)
    x_conc = tf.concat([d, conc], -1)

    # Upsample 2st
    x = Lambda(subpixel_conv2d(scale=2))(x_conc)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(subpixel_conv2d(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = a = upsample_1(x, 2, name='conv2d_1_scale_2')

        a = Conv2D(num_filters, 3, padding='same', activation='relu')(a)
        a = Conv2D(num_filters, 3, padding='same')(a)
        a = Add()([a, x])

        b = Conv2D(num_filters, 3, padding='same', activation='relu')(a)
        b = Conv2D(num_filters, 3, padding='same')(b)
        b = Add()([a, b])
        # b = Add()([b, x])

        c = Conv2D(num_filters, 3, padding='same', activation='relu')(b)
        c = Conv2D(num_filters, 3, padding='same')(c)
        c = Add()([b, c])
        # c = Add()([c, x])

        d = Conv2D(num_filters, 3, padding='same', activation='relu')(c)
        d = Conv2D(num_filters, 3, padding='same')(d)
        d = Add()([c, d])
        # d = Add()([d, x])

        e = Conv2D(num_filters, 3, padding='same', activation='relu')(d)
        e = Conv2D(num_filters, 3, padding='same')(e)
        e = Add()([d, e])

        f = Conv2D(num_filters, 3, padding='same', activation='relu')(e)
        f = Conv2D(num_filters, 3, padding='same')(f)
        f = Add()([e, f])

        g = Conv2D(num_filters, 3, padding='same', activation='relu')(f)
        g = Conv2D(num_filters, 3, padding='same')(g)
        g = Add()([f, g])

        h = Conv2D(num_filters, 3, padding='same', activation='relu')(g)
        h = Conv2D(num_filters, 3, padding='same')(h)
        h = Add()([g, h])

        # i = Conv2D(num_filters, 3, padding='same', activation='relu')(h)
        # i = Conv2D(num_filters, 3, padding='same')(i)
        # i = Add()([h, i])
        #
        # j = Conv2D(num_filters, 3, padding='same', activation='relu')(i)
        # j = Conv2D(num_filters, 3, padding='same')(j)
        # j = Add()([i, j])

        k = Conv2D(num_filters, 3, padding='same')(h)
        x = Add()([k, x])

        # for i in range(2):
        #     c = res_block(c, num_filters, 0)
        # x = Conv2D(num_filters, 3, padding='same')(c)
        # # x = Add()([x, c])

        x = upsample_1(x, 2, name='conv2d_2_scale_2')
        # x = Conv2D(num_filters, 3, padding='same')(x)

    return x
