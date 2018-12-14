import matplotlib.pyplot as plt
import openslide
import numpy as np
import billUtils as bu

openslide.Image.MAX_IMAGE_PIXELS = None

#def process_key(event):
#    print("Key:", event.key)
#
#def process_button(event):
#    print("Button:", event.x, event.y, event.xdata, event.ydata, event.button)
#
#

dir0 = '/media/roger/Seagate Expansion Drive/18-040_n7/HE'


tissue_file = bu.uichoosefile(title='Please choose a tissue file...', initialdir=dir0)
cpd_file = tissue_file.split(sep='.')
cpd_file[-2] += '_cpd'
cpd_file = '.'.join(cpd_file)

T = openslide.open_slide(tissue_file)
C = openslide.open_slide(cpd_file)

img0 = np.asarray(T.read_region((0,0), 0, T.dimensions))
img1 = np.asarray(C.read_region((0,0), 0, C.dimensions))

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

ax[0].imshow(img0)
ax[0].set_title(bu.just_filename(bu,tissue_file))
ax[1].imshow(img1)


#fig.canvas.mpl_connect('key_press_event', process_key)
#fig.canvas.mpl_connect('button_press_event', process_button)
#fig.canvas.mpl_connect(')

plt.show()

