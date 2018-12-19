import matplotlib.pyplot as plt
import openslide
import numpy as np
import billUtils as bu
import imageio

openslide.Image.MAX_IMAGE_PIXELS = None
TISSUE_ROOT_DIR_FILE = 'tissue_root_dir.txt'


#
#

try:
    with open(TISSUE_ROOT_DIR_FILE,'r') as f:
        dir0 = f.read()
        if dir0[-1] is '\n':
            dir0 = dir0[0:-1] # clip off the newline, as it confuses uichoosefile
        print('Starting at',dir0)
except:
    dir0 = bu.uichoosedir(title='Please choose a starting directory for tissue files...')
    with open(TISSUE_ROOT_DIR_FILE,'w') as f:
        if dir0:
            f.write(dir0)

tissue_file = bu.uichoosefile(title='Please choose a tissue file...', initialdir=dir0)
if not tissue_file:
    print('Ok, bye!')
else:
    cpd_file = tissue_file.split(sep='.')
    cpd_file[-2] += '_cpd'
    cpd_file = '.'.join(cpd_file)
    
    C = None
    T = openslide.open_slide(tissue_file)
    
    try:
        img1 = imageio.imread(cpd_file)
    except:
        print('oops no cpd')
    
    img0 = np.asarray(T.read_region((0,0), 0, T.dimensions))
    if img1.any():
        def process_key(event):
            print("Key:", event.key)
        
        def process_button(event):
            if event.inaxes:
                for a in ax:
                    a.plot(event.xdata, event.ydata, color='orange', marker='o')
                    a.figure.canvas.draw_idle()   
                    
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

        fig.canvas.mpl_connect('key_press_event', process_key)
        fig.canvas.mpl_connect('button_press_event', process_button)


        ax[0].imshow(img0)
        ax[0].set_title(bu.just_filename(bu,tissue_file))
        ax[1].imshow(img1)
    else:
        plt.imshow(img0)
        plt.title(bu.just_filename(bu,tissue_file))
    
    
    
    
    plt.show()
    
