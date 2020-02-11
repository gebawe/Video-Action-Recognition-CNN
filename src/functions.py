# -*- coding: utf-8 -*-
from __future__ import print_function
import os, shlex, struct, platform, subprocess, sys, shutil, numpy as np, cv2, matplotlib.pyplot as plt, math, mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
print('Python version : ', platform.python_version())
print('OpenCV version  : ', cv2.__version__)


def print_and_log(message, log=None):
    print(message)
    if log is not None: log.info(message)

#######################################################################
################ Progression bar in the terminal ######################
#######################################################################
def progress_bar(count, total, title, completed=0, log=None):
    terminal_size = get_terminal_size()
    percentage = int(100.0 * count / total)
    length_bar = max([4, terminal_size[0] - len(title) - len(str(total)) - len(str(count)) - len(str(percentage)) - 10])
    filled_len = int(length_bar * count / total)
    bar = '█' * filled_len + ' ' * (length_bar - filled_len)
    sys.stdout.write('%s [%s] %s %% (%d/%d)\r' % (title, bar, percentage, count, total))
    sys.stdout.flush()
    if completed:
        sys.stdout.write("\n")
        if log is not None:
            log.info('%s [%s] %s %% (%d/%d)' % (title, bar, percentage, count, total))

###############################################################################
######################### Files processing functions ##########################
###############################################################################
def make_new_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def make_path(path) :
    if not os.path.exists(path):
        os.mkdir(path)

def remove_file(path):
    try:
        os.remove(path)
    except OSError as e:
        print('Attention: Error n%s: file %s - %s' % (e.errno, e.filename, e.strerror))


###############################################################################
#################### Terminal size for different platform #####################
###############################################################################
def get_terminal_size():
    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os in ['Linux', 'Darwin'] or current_os.startswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)      # default value
    return tuple_xy

def _get_terminal_size_windows():
    try:
        from ctypes import windll, create_string_buffer
        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom,
             maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
            return sizex, sizey
    except:
        pass

def _get_terminal_size_tput():
    try:
        cols = int(subprocess.check_call(shlex.split('tput cols')))
        rows = int(subprocess.check_call(shlex.split('tput lines')))
        return (cols, rows)
    except:
        pass

def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios
            cr = struct.unpack('hh',
                               fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            return cr
        except:
            pass
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (os.environ['LINES'], os.environ['COLUMNS'])
        except:
            return None
    return int(cr[1]), int(cr[0])


##################################################################
######################### Satat dataset ##########################
##################################################################
def stat_dataset(path = 'Data', log = None):
    actions = [action for action in os.listdir(path) if os.path.isdir(os.path.join(path,action))]; nb_actions = []

    for action in actions:
        videos = os.listdir(os.path.join(path, action))
        nb_frames = []
        for video in videos:
            total_frames = len(os.listdir(os.path.join(path, action, video, 'RGB')))
            nb_frames.append(total_frames)
        nb_actions.append(nb_frames)
        print_and_log('%d actions for %s with %d frames +- %d (max = %d, min = %d)'
              % (len(videos), action, np.mean(nb_frames), np.std(nb_frames), max(nb_frames), min(nb_frames)), log = log)

    flat_nb_actions = [nb_frame for list_nb_frame in nb_actions for nb_frame in list_nb_frame]
    print_and_log('\nTotal : %d videos for %s actions with ~%d frames +- %d (max = %d, min = %d)'
                  % (len(flat_nb_actions), len(actions), np.mean(flat_nb_actions), np.std(flat_nb_actions), max(flat_nb_actions), min(flat_nb_actions)),
                  log = log)
    return 1

def count_frames_manual(video):
	total = 0
	while True:
		(grabbed, frame) = video.read()
		if not grabbed:
			break
		total += 1
	return total

###################################################################
#################### Functions to show images #####################
###################################################################
def ShowImage(Image, title='Show Image', Scalling=1, wait=1):
    resized_image = cv2.resize(Image, (Image.shape[1]/Scalling, Image.shape[0]/Scalling))
    cv2.imshow(title, resized_image)
    if wait >= 0:
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            sys.exit('Program stoped')

###########################
#####        #        #####
##### image1 # image2 #####
#####        #        #####
###########################
#####        #        #####
##### image3 # image4 #####
#####        #        #####
###########################
def JoinImages(image1, image2, image3, image4):
    if(len(image1.shape)<3):
        image1 = np.dstack((image1, image1, image1))
    if(len(image2.shape)<3):
        image2 = np.dstack((image2, image2, image2))
    if(len(image3.shape)<3):
        image3 = np.dstack((image3, image3, image3))
    if(len(image4.shape)<3):
        image4 = np.dstack((image4, image4, image4))
    up = np.concatenate((image1, image2), axis=1) # /image2.max()
    left = np.zeros(image1.shape)
    right = np.zeros(image2.shape)
    left[int(image1.shape[0]/2 - image3.shape[0]/2) : int(image1.shape[0]/2 - image3.shape[0]/2) + image3.shape[0],\
        int(image1.shape[1]/2 - image3.shape[1]/2) : int(image1.shape[1]/2 - image3.shape[1]/2) + image3.shape[1]] = image3
    right[int(image2.shape[0]/2 - image4.shape[0]/2) : int(image2.shape[0]/2 - image4.shape[0]/2) + image4.shape[0],\
        int(image2.shape[1]/2 - image4.shape[1]/2) : int(image2.shape[1]/2 - image4.shape[1]/2) + image4.shape[1]] = image4 # /image4.max()
    down = np.concatenate((left, right), axis=1)
    return np.concatenate((up, down), axis=0).astype('uint8')

def draw_hsv(flow, normalization=30, inv=True):

    fy = flow[:,:,1]; fx = flow[:,:,0]
    h, w = fx.shape
    ang = np.arctan2(fy, fx) + np.pi
    v = normalization*np.sqrt(fx*fx + fy*fy)/math.sqrt(2)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if inv:
        visualization = 255*np.ones(visualization.shape, dtype=np.uint8) - visualization
    return visualization.astype('uint8')

def draw_flow(img, flow, step=10):
    vis = img.copy()
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


###################################################################
################# Spatial Segmentation functions ##################
###################################################################
def find_roi(amplitude, size_data, alpha = 0.5):
    # Crop based on the maximum value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(amplitude)
    if maxVal == 0 :
        x = amplitude.shape[1]/2; y = amplitude.shape[0]/2
        return x, y
    else:
        x = maxLoc[0]; y = maxLoc[1]
        x, y = correction_coordinates(x, y, size_data, amplitude.shape)

    # Crop based on the concentration of points
    index = cv2.findNonZero(cv2.convertScaleAbs(amplitude))

    if index is None:
        x2 = amplitude.shape[1]/2; y2 = amplitude.shape[0]/2
    else:
        index = np.mean(index, axis=0)
        if len(index)==1:
            x2 = int(index[0][0]); y2 = int(index[0][1])
        else:
            x2 = int(index[0]); y2 = int(index[1])
        x2, y2 = correction_coordinates(x2, y2, size_data, amplitude.shape)

    # Fusion
    return alpha * x + (1 - alpha) * x2, alpha * y + (1 - alpha) * y2

def correction_coordinates(x, y, size, shape):
    if x < int(size[1] * 0.5): x = int(size[1] * 0.5)
    if x >  int(shape[1] - size[1] * 0.5): x = int(shape[1] - size[1] * 0.5)
    if y < int(size[0] * 0.5): y = int(size[0] * 0.5)
    if y > int(shape[0] - size[0] * 0.5): y = int(shape[0] - size[0] * 0.5)
    return x, y


###################################################################
############################ Figures ##############################
###################################################################
def make_train_figure(loss_train, loss_val, acc_val, acc_train, title):
    host = host_subplot(111, axes_class=AA.Axes)
    par = host.twinx()

    host.set_xlabel("Epochs")
    host.set_ylabel("Loss")
    par.set_ylabel("Accuracy")

    par.axis["right"].toggle(all=True)

    epochs = [i for i in range(1, len(loss_val)+1)]

    host.set_xlim(1, len(epochs))
    host.set_ylim(0, np.max([np.max(loss_train), np.max(loss_val)]))
    par.set_ylim(0, 1)

    max_acc = max(acc_val)
    max_acc_idx = epochs[acc_val.index(max_acc)]
    host.set_title("Max Validation Accuracy: %.1f%% at iteration %d" % (max_acc*100, max_acc_idx))

    host.plot(epochs, loss_train, label="Train loss", linewidth=1.5)
    host.plot(epochs, loss_val, label="Validation loss", linewidth=1.5)
    par.plot(epochs, acc_val, label="Validation Accuracy", linewidth=1.5)
    par.plot(epochs, acc_train, label="Train Accuracy", linewidth=1.5)

    host.legend(loc='lower right', ncol=1, fancybox=False, shadow=True)

    plt.savefig(title)
    plt.close('all')


def plot_histogram(data, nb_of_columns, range_lim=None, title='Figure', ylabel='y', xlabel='x', color='b', grid=True, labels=None, show=False, save_path=None, figsize=(5,3.5)):

    plt.figure(figsize=figsize)
    if labels is not None:
        if range_lim is None: range_lim=[np.asarray([i.min() for i in data]).min(), np.asarray([i.max() for i in data]).max()]
        for idx, label in enumerate(labels): plt.hist(data[idx], nb_of_columns, range_lim, facecolor=color[idx], alpha=0.5, label=label)
        plt.legend()
    else:
        if range_lim is None: range_lim=[data.min(), data.max()]
        plt.hist(data, nb_of_columns, range_lim, facecolor=color, alpha=0.75)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)
    plt.tight_layout()
    if save_path is not None: plt.savefig(save_path)
    if show: plt.show()
    plt.close('all')


###################################################################
####################### Error images function #####################
###################################################################
def computeErrorImage(im1, im2):
    res = cv2.addWeighted(im1, 1, im2, -1, 128)
    return res

def computeGME(flow):
    src = np.zeros_like(flow)
    h, w = flow.shape[:2]
    c = np.array([w/2, h/2])
    src[:,:,0] += np.arange(w)
    src[:,:,1] += np.arange(h)[:,np.newaxis]
    src -= c

    dst = src + flow

    srcPts = src.reshape((h*w, 2))
    dstPts = dst.reshape((h*w, 2))

    hom, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)
    dst2 = cv2.perspectiveTransform(src, hom)

    gme = dst2 - src

    return gme

def computeGMEError(flow, gme):
    err = np.sqrt(np.square(flow[:,:,0] - gme[:,:,0]) + np.square(flow[:,:,1] - gme[:,:,1]))
    return err

def computeCompensatedFrame(prev, flow):
    h, w = flow.shape[:2]
    map = flow.copy()
    #map = -map  #B?
    map[:,:,0] += np.arange(w)
    map[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(prev, map, None, cv2.INTER_LINEAR)
    return res

def computeMSE(prev, curr):
    m = cv2.absdiff(prev, curr)
    m = m.astype(np.int32)
    m = np.square(m)
    s = m.sum() / m.size
    return s

def computePSNR(mse):
    if (mse > 0):
        return 10 * np.log10((255*255) / mse)
    else:
        return 0

def computeEntropy(img):
    h, w = img.shape[:2]
    hist, bin_edges = np.histogram(img, bins=255, range=(0, 255))
    hist = hist.astype(np.float32) / (w*h)
    loghist = np.log2(np.array([1 if x==0 else x for x in hist]))
    m = hist*loghist
    ent = - m.sum()
    return ent
