import matplotlib.pyplot as plt
import openslide
import numpy as np
import billUtils as bu
import imageio.v2 as imageio
from skimage import morphology
from scipy.ndimage import binary_fill_holes

openslide.Image.MAX_IMAGE_PIXELS = None
TISSUE_ROOT_DIR_FILE = 'tissue_root_dir.txt'
THRESHOLD = float(230/255)

#
# Choose a *_annotated.tif file as the "tissue file".

class TandemViewer():
    def __init__(self):

        try:
            with open(TISSUE_ROOT_DIR_FILE,'r') as f:
                dir0 = f.read()
                if dir0[-1] == '\n':
                    dir0 = dir0[0:-1] # clip off the newline, as it confuses
                                      #  uichoosefile
                print('Starting at',dir0)
        except:
            dir0 = bu.uichoosedir(title='Please choose a starting directory for tissue files...')
            with open(TISSUE_ROOT_DIR_FILE,'w') as f:
                if dir0:
                    f.write(dir0)
        
        tissue_file = bu.uichoosefile(title='Please choose a tissue file...', \
                                      initialdir=dir0)
        if not tissue_file:
            print('Ok, bye!')
        else:
            self.press = False
            self.move = False

            test4 = None
            if 'test4' in tissue_file:
                tparts = tissue_file.split(sep='/')[0:-1]
                tparts.append('test4_cancer.tif')
                test4 = '/'.join(tparts)
                print('test4 is', test4)
                
            cpd_file = tissue_file.split(sep='.')
            cpd_file[-2] += '_cpd'
            cpd_file = '.'.join(cpd_file)
            
#            C = None   # openslide can't read single channel TIFF
            T = openslide.open_slide(tissue_file)
            
            try:
                img1 = imageio.imread(cpd_file)
            except:
                print('oops no cpd')
            
            img0 = np.asarray(T.read_region((0,0), 0, T.dimensions))
            if img1.any():
                
#                ck = self.circle_kernel(13)
                            
                if not test4:
                    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
                else:
                    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
                self.ax = ax
                
                fig.canvas.mpl_connect('key_press_event', self.process_key)
                
                fig.canvas.mpl_connect('button_release_event',\
                                       self.process_button_release)
                fig.canvas.mpl_connect('button_press_event',\
                                       self.process_button)
                fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        
        
                ax[0].imshow(img0)
                ax[0].set_title(bu.just_filename(bu,tissue_file))
                show1 = binary_fill_holes(img1/np.max(img1) > THRESHOLD)
#                show1 = morphology.erosion(show1, selem=ck)
                
                ax[1].imshow(show1)
                ax[1].set_title('model output')
                if test4:
                    ax[2].imshow(imageio.imread(test4))
                    ax[2].set_title('human pathologist')
            else:
                plt.imshow(img0)
                plt.title(bu.just_filename(bu,tissue_file))
    
    

    def circle_kernel(self, N):
        if N % 2 != 1:
            print('kernel size must be odd...')
            return None
        
        rmax = (N-1)/2.0
        x = np.linspace(-rmax,rmax,num=N)
        xx = np.reshape(np.kron(np.power(x,2),np.ones_like(x)),(N,N))
        r = np.sqrt(xx + np.transpose(xx))
        ck = r < rmax
        ck = ck/np.sum(ck)
    
        return ck
    
    def process_key(self, event):
        print("Key:", event.key)

    def process_button(self, event):
#        print('in axes is',event.inaxes)
        if event.inaxes:
            self.press = True
            
    def process_button_release(self, event):
        if event.inaxes and not self.just_dragged():
            for a in self.ax:
                a.plot(event.xdata, event.ydata, markerfacecolor='#ff6700',\
                       markeredgecolor = '#ccff00', marker='D', \
                       markersize = 7)
                a.figure.canvas.draw_idle() 
        self.press = False
        self.move = False

    def on_move(self, event):
        if self.press:
            self.move=True
    
    def just_dragged(self):
#        print('press is',self.press,': move is ',self.move)
        return self.press and self.move
        
        

if __name__=="__main__":
    tv = TandemViewer()
    
    plt.show()
    
