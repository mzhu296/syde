"""Click corresponding points in two images and save them to a .json file.

    python pick_points.py data/IMG_5176.jpg data/IMG_5178.jpg points/set1.json

Click alternately: left image, right image, left, right, ... Close the window
(or press Enter) when done. For rectification pass the same image only once.
"""
import json
import sys
import matplotlib.pyplot as plt
import imageio.v3 as iio


def main(paths, out):
    ims = [iio.imread(p) for p in paths]
    fig, axs = plt.subplots(1, len(ims), figsize=(8 * len(ims), 6), squeeze=False)
    for ax, im in zip(axs[0], ims):
        ax.imshow(im)
    pts = [[] for _ in ims]

    def onclick(e):
        for k, ax in enumerate(axs[0]):
            if e.inaxes is ax and e.button == 1:
                pts[k].append([float(e.xdata), float(e.ydata)])
                ax.plot(e.xdata, e.ydata, 'r+', ms=14)
                ax.text(e.xdata + 6, e.ydata - 6, str(len(pts[k])), color='yellow')
                fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    with open(out, 'w') as f:
        json.dump({'images': paths, 'points': pts}, f, indent=1)
    print('saved', out, [len(p) for p in pts])


if __name__ == '__main__':
    main(sys.argv[1:-1], sys.argv[-1])
