#!/usr/bin/env python
# encoding: utf-8

"""
This code builds on some functions provided by Ben Southgate on his
blog post <http://bensouthgate.com/p/12_3_13.php>_"Constructing Color Gradients with Python"  I couldn't get his permission, becuase he doesn't leave his contact information anywhere.  If you see him, tell him I'm using it and make sure it's ok.


"""

import numpy
import pandas


def hex_to_RGB(hex):
    """
    Convert hex color to RGB, i.e. "#FFFFFF" -> [255,255,255] for either a
    single :class:`str` hex value or a :class:`pandas.Series` of hex values

    """

    # Pass 16 to the integer function for change of base
    def _hex_to_RGB(hex):
        return pandas.Series([int(hex[i:i + 2], 16) for i in range(1, 6, 2)],
                             index=['r', 'g', 'b'])

    if isinstance(hex, str):
        return _hex_to_RGB(hex)
    else:
        return hex.apply(_hex_to_RGB)


def RGB_to_hex(RGB):
    """
    Convert RGB color to hex, i.e. [255,255,255] -> "#FFFFFF" for either a
    single :class:`str` hex value or a :class:`pandas.Series` of hex values

    :NOTE: RGB values *must* be of :type:`int`
    """

    def _RGB_to_hex(RGB):
        return "#" + "".join(["0{0:x}".format(v) if v < 16 else
                              "{0:x}".format(v) for v in RGB])

    if isinstance(RGB, pandas.DataFrame):
        return RGB.apply(_RGB_to_hex, axis=1)
    else:
        return _RGB_to_hex(RGB)


def pd_linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    """
    A linear (two-stop) gradient creator that creates n RGB (including the
    endpoints) betwene the two hexadecimal values

    :ARGS:

        start_hex: :class:`str` of the starting hex color code (ex. '#7195a3')

        finish_hex: :class:`str`of the ending hex color code

        n: :class:`int` the number of stops between `start_hex` and `finish_hex`

    :RETURNS:

        :class:`pandas.DataFrame` with columns ['r','g','b'], and index 0, n

    """
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)

    lin_grad = []
    for i in range(len(s)):
        lin_grad.append(numpy.linspace(s[i], f[i], n))

    return pandas.DataFrame(numpy.rint(lin_grad).transpose().astype(int),
                            columns=['r', 'g', 'b'])


def three_stop_gradient(start_hex, finish_hex, mid_hex='#FFFFFF', n=20):
    """
    Create a three stop gradient from start_hex, to mid_hex, to finish_hex (which is
    defalut white), with n stops

    :ARGS:

        start_hex: :class:`str` of the starting hex color

        finish_hex :class:`str` of th ending hex color

        n: :class:`int` the number of stops between the start, mid, and finish hex
    """
    first_grad = pd_linear_gradient(start_hex, mid_hex, n=n / 2 + 1).iloc[:-1, :]
    second_grad = pd_linear_gradient(mid_hex, finish_hex, n=n / 2 + 1).iloc[1:, :]
    return first_grad.append(second_grad)
