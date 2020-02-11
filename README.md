# Implementation-of-CNN-for-action-recognition-in-videos
Deep Learning in Computer Vision to build a convolutional neural network with the Keras framework in Python aimed at recognizing actions in videos.  The UCF101 dataset, which is formed by YouTube videos gathered in 101 categories is used to train and test the proposed approach.
A siamese convolutional network has ben developed formed by two branches that uses the same weights Each branch has the same architecture while working in tandem on two different input vectors to compute comparable output vectors. One of the branches will correspond to the actual frame of the video (RGB frame), while the other one will correspond to its optical flow. In this way, by checking both characteristics at the same time and merging them, we will obtain the probabilities of the video belonging to each of the 101 classes.

## mplementation-of-CNN-for-action-recognition-in-videos

Author: Awet Haileslassie Gebrehiwot(awethaileslassie21@gmail.com)

If you have any questions on the code or README, please feel free to contact me.

### Requirments
python 
Tensorflow
Numpy
Matplotlib

### Running the code 
first download the UCF101 dataset  and start the training

Open terminal and navigate to the directory where the code exist
then simply type:

python skeleten.py


### Description

The code provides a predefined model.
The  model is CNN (convolutional neural networks).

![dataset_after_change_1_if_image_represents_number_3_and_0_otherwise.png](https://github.com/awethaileslassie/awet_mnist_pytorch/blob/master/dataset_after_change_"1"_if_image_represents_number_3_and_"0"_otherwise.png)
The network is trained and evaluated on MNIST dataset with classification accuracy. 


#### Part 3: CNN model

The siams CNN model contains two Baranches
One of the branches will correspond to the actual frame of the video (RGB frame), while the other one will correspond to its optical flow. then both will be conacatnated to result in better performance for action recognition of the 101 classes.

1: Extract RGB 
The 1st thing to do is to obtain RGB content for each video sequence. For doing so, we use from the provided
code the functions extract RGB, where we can extract RGB information from the video content.
2: Calculating optical flow
The 2nd thing to do is to compute flow for each video sequence. For doing so, we use from the provided code the
functions compute flow, where we can calculate and compute the optical flow from the video.

# Architecture
![Loss_50_epoch.png](https://github.com/awethaileslassie/awet_mnist_pytorch/blob/master/Loss_50_epoch.png)


# Model Loss during Training for 30 epoches 
![5_3-loss.png](https://github.com/awethaileslassie/Implementation-of-CNN-for-action-recognition-in-videos/blob/master/DLCV5_Graphs/5_3-valloss.png)

# Model Accuracy during Training for 30 epoches
![5_3-acc.png](https://github.com/awethaileslassie/awet_mnist_pytorch/blob/master/Accuracy_50_epoch.png)


# Model Loss during Testing for 30 epoches 
![5_3-valloss.png](https://github.com/awethaileslassie/Implementation-of-CNN-for-action-recognition-in-videos/blob/master/DLCV5_Graphs/5_3-valloss.png)

# Model Accuracy during Testing for 30 epoches
![5_3-valacc.png](https://github.com/awethaileslassie/awet_mnist_pytorch/blob/master/Accuracy_50_epoch.png)
