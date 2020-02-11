import numpy as np
import keras
import os, time, random
from functions import *

#####################################################################
######################### Frame extraction ##########################
#####################################################################
def extract_RGB(data_path = 'Videos', output_path = 'Data', height = 120):
    action_list = os.listdir(data_path)
    num_videos = len([f for action in action_list for f in os.listdir(os.path.join(data_path, action)) if os.path.isfile(os.path.join(data_path, action, f))])

    make_path(output_path)
    
    # Chrono
    start_time = time.time(); count = 0

    for action in action_list:
        make_path(os.path.join(output_path, action))
        video_list = os.listdir(os.path.join(data_path, action))
        for video in video_list :
            progress_bar(count, num_videos, 'Frame extraction - %s' % action)
            make_path(os.path.join(output_path, action, video))
            make_new_path(os.path.join(output_path, action, video, 'RGB'))
            os.system("ffmpeg -hide_banner -loglevel error -i %s -vf scale=-1:%d -start_number 0 %s" %
                       (os.path.join(data_path, action, video), height, os.path.join(output_path, action, video, 'RGB', '%08d.png')))
            count+=1
    progress_bar(count, num_videos, 'Frame extraction completed in %d s' % int(time.time() - start_time), 1)
    return 1


#############################################################################
######################### Optical Flow computation ##########################
#############################################################################
def compute_flow(data_path = 'Data', flow_calculation = 0, size_data = [100,100], visualization = False):
    action_list = os.listdir(data_path)
    num_videos = len([f for action in action_list for f in os.listdir(os.path.join(data_path, action))])
    max_flow_values = []

    # Chrono
    start_time = time.time(); count = 0

    for action in action_list:
        video_list = os.listdir(os.path.join(data_path, action))
        for video in video_list :
            progress_bar(count, num_videos, 'Flow computation - %s - %s' % (action, video[-11:-4])); count += 1
            Video_path = os.path.join(data_path, action, video)
            RGB_path = os.path.join(Video_path, 'RGB'); Flow_path = os.path.join(Video_path, 'DeepFlowOpenCV')
            length_video = len([f for f in os.listdir(RGB_path) if os.path.isfile(os.path.join(RGB_path, f))])
            x_list = []; y_list = []

            if flow_calculation == 1:
                make_new_path(Flow_path)
                DeepOpenCV = cv2.optflow.createOptFlow_DeepFlow()
                old_frame = cv2.imread(os.path.join(RGB_path, '%08d.png' % 0), 0)

            for frame_number in range(1, length_video):

                # Flow calculation
                if flow_calculation == 1:
                    new_frame = cv2.imread(os.path.join(RGB_path, '%08d.png' % frame_number), 0)
                    flow = DeepOpenCV.calc(old_frame, new_frame, None)
                    # Residual Flow
                    flow = flow - computeGME(flow)
                    old_frame = new_frame.copy()
                    np.save(os.path.join(Flow_path, '%08d' % frame_number), flow)
                else:
                    flow = np.load(os.path.join(Flow_path, '%08d.npy' % frame_number))

                # Maximum value for the whole dataset
                max_flow_values.append(abs(flow).max())

                # Spatial segmentation
                amplitude = flow*flow; amplitude = np.sqrt(amplitude[:,:,0] + amplitude[:,:,1])
                x, y = find_roi(amplitude, size_data)
                x_list.append(x); y_list.append(y)

                if visualization:
                    x = int(x - size_data[1] * 0.5); y = int(y - size_data[0] * 0.5)
                    rgb = cv2.imread(os.path.join(RGB_path, '%08d.png' % frame_number))
                    flow_vis = draw_hsv(flow)
                    ShowImage(rgb, title='RGB'); ShowImage(draw_flow(rgb, flow), title='RGB flow')
                    ShowImage(flow_vis, title='Flow'); ShowImage(amplitude, title='Amplitude')
                    ShowImage(rgb[y : y + size_data[0], x : x + size_data[1]], title='RGB segmented')
                    ShowImage(flow_vis[y : y + size_data[0], x : x + size_data[1]], title='Spatial segmentation flow', wait=0)

            # Smoothing of the spatial segmentation
            x_list = cv2.GaussianBlur(np.asarray(x_list), (1, int(2 * 20 + 1)), 0)
            y_list = cv2.GaussianBlur(np.asarray(y_list), (1, int(2 * 20 + 1)), 0)

            # Save spatial segmentation
            np.save(os.path.join(Video_path, 'spatial_segmentation'), [x_list, y_list])
    # Save Max Flow values
    np.save(os.path.join(data_path, 'max_flow_values'), max_flow_values)
    progress_bar(count, num_videos, 'Flow computation completed in %d s' % int(time.time() - start_time), 1)
    return 1

def get_flow_normalization(flow_values_path = '/espace/DLCV2/Data/max_flow_values.npy', visualization = 0, show = False, method = '99.99%'):
    if method not in ['Max', '99.99%', 'Normal']: method = '99.99%'
    flow_values = np.load(flow_values_path)
    max_flow = max(flow_values); mean_flow = np.mean(flow_values); std_flow = np.std(flow_values)
    if method=='Max': # Max Normalization
        norm_flow = max_flow
    elif method=='99.99%': # Max Normalization
        norm_flow = sorted(flow_values)[int(0.9999*len(flow_values))]
    elif method=='Normal': # Normal Normalization
        norm_flow = np.mean(flow_values) + 3*np.std(flow_values)
    print('Flow : Max %.2f - Mean %.2f +- %.2f\nNormalization method : %s, flow divided by %.2f' % (max_flow, mean_flow, std_flow, method, norm_flow))
    if visualization:
        make_new_path('Visualization')
        plot_histogram(flow_values, 500, title='Histogram of Max Optical Flow values', save_path = 'Visualization/histo_max_flow.png', show = show)

    return norm_flow


#############################################################################
######################### Define Train, Val, Test ###########################
#############################################################################
def define_list(train_file, test_file, class_file):
    file = open(train_file, "r")
    train_list =[line.split()[0] for line in file]
    file.close()

    file = open(test_file, "r")
    test_list =[line.split()[0] for line in file]
    file.close()

    file = open(class_file, "r")
    class_index =[line.split()[1] for line in file]
    file.close()

    return class_index, train_list, test_list

#############################################
############## Data Generator ###############
#############################################
class My_data_generator():
    def __init__(self, dataset_list, flow_normalization, class_index, batch_size, augmentation=False):
        self.dataset_list = dataset_list; self.flow_normalization = flow_normalization
        self.class_index = class_index; self.batch_size = batch_size
        self.augmentation = augmentation; self.iter_initiated = 0
        self.max = int(math.ceil(len(dataset_list)/float(batch_size))); iter(self)

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0; random.shuffle(self.dataset_list)
        return self

    def next(self):
        if self.n >= len(self.dataset_list): iter(self)
        rgb_batch = []; flow_batch = []; label_batch = []
        index_batch = self.n
        self.n += self.batch_size
        for index in range(index_batch, min([index_batch + self.batch_size, len(self.dataset_list)])):
            rgb, flow, label = get_data(self.dataset_list[index], self.flow_normalization, self.class_index, augmentation = self.augmentation)
            rgb_batch.append(rgb); flow_batch.append(flow); label_batch.append(label)
        return ([np.array(rgb_batch), np.array(flow_batch)], np.array(label_batch))

class My_Data_Sequence(keras.utils.Sequence):
    def __init__(self, dataset_list, flow_normalization, class_index, batch_size, augmentation=False, shuffle=True):
        self.dataset_list = dataset_list; self.flow_normalization = flow_normalization
        self.class_index = class_index; self.batch_size = batch_size
        self.augmentation = augmentation; self.shuffle = shuffle
        if self.shuffle: random.shuffle(self.dataset_list)

    def __len__(self):
        return int(np.ceil(len(self.dataset_list)/float(self.batch_size)))

    def __getitem__(self, idx):
        rgb_batch = []; flow_batch = []; label_batch = []
        for index in range(idx * self.batch_size, min([(idx + 1) * self.batch_size, len(self.dataset_list)])):
            rgb, flow, label = get_data(self.dataset_list[index], self.flow_normalization, self.class_index, augmentation = self.augmentation)
            rgb_batch.append(rgb); flow_batch.append(flow); label_batch.append(label)
        return ([np.array(rgb_batch), np.array(flow_batch)], np.array(label_batch))

    def on_epoch_end(self):
        if self.shuffle: random.shuffle(self.dataset_list)


class My_Data_Sequence_one_branch(keras.utils.Sequence):
    def __init__(self, dataset_list, flow_normalization, class_index, batch_size, data_type='RGB', augmentation=False, shuffle=True):
        self.dataset_list = dataset_list; self.flow_normalization = flow_normalization
        self.class_index = class_index; self.batch_size = batch_size
        self.augmentation = augmentation; self.shuffle = shuffle
        self.data_type = data_type
        if self.shuffle: random.shuffle(self.dataset_list)

    def __len__(self):
        return int(np.ceil(len(self.dataset_list)/float(self.batch_size)))

    def __getitem__(self, idx):
        data_batch = []; label_batch = []
        for index in range(idx * self.batch_size, min([(idx + 1) * self.batch_size, len(self.dataset_list)])):
            rgb, flow, label = get_data(self.dataset_list[index], self.flow_normalization, self.class_index, augmentation = self.augmentation)
            if self.data_type == 'RGB':
                data_batch.append(rgb)
            else:
                data_batch.append(flow)
            label_batch.append(label)
        return (np.array(data_batch), np.array(label_batch))

    def on_epoch_end(self):
        if self.shuffle: random.shuffle(self.dataset_list)


#############################################################################
######################### Extract the Data from path ########################
#############################################################################
def get_data(action_path, flow_normalization, class_index, data_path = '/espace/DLCV2/Data', augmentation = 0,
             size_data = [100,100], dt = 100, save_option = 0, save_path = 'Visualization', show_option = 0):
    # Variables
    Video_path = os.path.join(data_path, action_path); label = action_path.split('/')[0]
    RGB_path = os.path.join(Video_path, 'RGB'); Flow_path = os.path.join(Video_path, 'DeepFlowOpenCV')
    length_video = len([f for f in os.listdir(RGB_path) if os.path.isfile(os.path.join(RGB_path, f))])
    x_list, y_list = np.load(os.path.join(Video_path, 'spatial_segmentation.npy'))
    rgb_data = []; flow_data = []; count = 0

    if augmentation: 
        angle, zoom, tx, ty, flip, start = augmentation_parameters(length_video, size_data, dt)
    else: 
        start = (length_video - dt)/2

    for frame_number in range(start, start + dt):
        if frame_number < 0: 
            frame_number = 0; flow_multiplicator = 0
        elif frame_number >= length_video: 
            frame_number = length_video - 1; flow_multiplicator = 0
        else: 
            flow_multiplicator = 1

        # Data Normalized
        rgb = cv2.imread(os.path.join(RGB_path, '%08d.png' % frame_number)).astype(float) / 255
        if frame_number == 0: frame_number = 1
        flow = flow_multiplicator * np.load(os.path.join(Flow_path, '%08d.npy' % frame_number)) / flow_normalization
        flow[flow > 1] = 1; flow[flow < -1] = -1

        x = x_list[frame_number - 1]; y = y_list[frame_number - 1]
        if augmentation:
            rgb, flow, x, y = apply_augmentation(rgb, flow, x, y, angle, zoom, tx, ty, flip, size_data)

        # Data segmented
        x = int(x - size_data[1] * 0.5); y = int(y - size_data[0] * 0.5)
        crop_rgb = rgb[y : y + size_data[0], x : x + size_data[1]]
        crop_flow = flow[y : y + size_data[0], x : x + size_data[1]]

        if save_option or show_option:
            data_image = JoinImages(255*rgb, draw_hsv(flow, normalization=255), 255*crop_rgb, draw_hsv(crop_flow, normalization=255))
            if show_option: ShowImage(data_image, title = 'Data Visualization')
            if save_option:
                make_path(os.path.join(save_path, label)); make_path(os.path.join(save_path, action_path))
                cv2.imwrite(os.path.join(save_path, action_path, '%08d.png' % count), data_image); count+=1

        rgb_data.append(cv2.split(crop_rgb)); flow_data.append(crop_flow)

    label_data = np.zeros(len(class_index)); label_data[class_index.index(label)] = 1
    rgb_data = np.transpose(rgb_data, (0, 2, 3, 1)); flow_data = np.array(flow_data)

    return rgb_data, flow_data, label_data


def augmentation_parameters(length_video, size_data, dt, rotation_range = 10, translation_range = 0.1, flip_option = True, zoom_range = 0.1):
    angle = (random.random()* 2 - 1) * rotation_range; zoom = 1 + (random.random()* 2 - 1) * zoom_range
    tx = random.randint(-translation_range * size_data[1], translation_range * size_data[1])
    ty = random.randint(-translation_range * size_data[0], translation_range * size_data[0])
    if flip_option: flip = random.randint(0,1)
    else: flip = 0

    # Normal distribution to pick starting point of the successive points
    start_range = length_video - dt
    mu = start_range/2; sigma = abs(start_range/6); start = None
    while not min([1, mu - 0.1*length_video]) <= start <= max([mu + 0.1*length_video, start_range]): start = int(np.random.normal(mu, sigma))

    return angle, zoom, tx, ty, flip, start

def apply_augmentation(rgb, flow, x, y, angle, zoom, tx, ty, flip, size_data):
    # Rotation Matrix
    R = cv2.getRotationMatrix2D((x, y), angle, 1)

    # Resize and apply transformations
    rgb = cv2.resize(cv2.warpAffine(rgb, R, (rgb.shape[1], rgb.shape[0])), (0,0), fx = zoom, fy = zoom)
    flow = cv2.resize(cv2.warpAffine(flow, R, (flow.shape[1], flow.shape[0])), (0,0), fx = zoom, fy = zoom)/zoom

    # Update Flow values according to rotation
    angle_radian = math.radians(angle)
    tmp = cv2.addWeighted(flow[:,:,0], math.cos(angle_radian), flow[:,:,1], -math.sin(angle_radian), 0)
    flow[:,:,1] = cv2.addWeighted(flow[:,:,0], math.sin(angle_radian), flow[:,:,1], math.cos(angle_radian), 0)
    flow[:,:,0] = tmp

    # Coordinates correction to fit in the image
    x, y = correction_coordinates(zoom * (x + tx), zoom * (y + ty), size_data, rgb.shape)

    if flip:
        rgb = cv2.flip(rgb, 1); flow = -cv2.flip(flow, 1); x = rgb.shape[1] - x; y = rgb.shape[0] - y

    return rgb, flow, x, y

