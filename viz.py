#!/usr/bin/env python
"""
File: viz
Date: 4/22/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""


import os
import sys
import numpy as np

from nibabel.testing import data_path
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

def listdir(directory):
    """
    Gets the full paths to the contents of a directory
    :param directory: A path to some directory
    :return: An iterator yielding full paths to all files in the specified directory
    """
    m = map(lambda d: os.path.join(directory, d), os.listdir(directory))
    contents = [f for f in m if not f.startswith('.')]
    return contents

def visualize(filename, data):
    FFMpegWriter = manimation.writers['ff mpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    im = plt.imshow(f(x, y), animated=True)

    def updatefig(*args):
        global x, y
        x += np.pi / 15.
        y += np.pi / 20.
        im.set_array(f(x, y))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()


    movie_fname = "%s.mp4" % os.path.splitext(filename)[0]
    with writer.saving(fig, movie_fname, 100):


def main():

    directory = sys.argv[1]
    for filename in listdir(directory)
        print("visualizing: %s ..." % filename)
        img = nib.load(filename)
        data = img.get_data()
        visualize(filename, data)

    print("Done.")

if __name__ == "__main__":
    main()