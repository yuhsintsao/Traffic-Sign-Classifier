# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./images/barchart.png "Visualization"
[image2]: ./images/visualization.png "Visualization"
[image3]: ./images/grayscale.png "Grayscaling"
[image4]: ./images/newimages.png "New Traffic Signs"
[image5]: ./images/newimages_gray.png "New Traffic Signs: Grayscale"
[image6]: ./images/confusionmatrix.png "Confusion Matrix"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799. 
* The size of the validation set is 4,410. 
* The size of test set is 12,630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed.

![Frequency of Classes][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the color images to grayscale because in that case the number of dimensions of input data is reduced from 3 to 1. Usually traffic signs are designed with a single color. Thus, color information of traffic signs can be ignored for the model to achieve better computing efficiency. Then I adopt histogram equalization to all images in order to increase the global contrast. Some images are with the target traffic sign and backgrounds both too dark. By adopting histogram equalization, we gain more details of the images. However, the trade-off is that some background noises are enhanced.

Here is an example of a traffic sign image before and after grayscaling.

![Color Image][image2]

![Grayscale Image][image3]

Without data normalization, the model achieve generally poor performance, i.e. a lower validation accuracy and a slower rate of convergence. Thus, batch normalization or global normalization is crucial. I found there is not much difference between two kinds, so I adopt global normalization only.  

I increase the size of dataset by adding transformed images. I rescaled and rotated all training images by randomly generating a series of angles between -30 and 30 degree. Here are examples of original images and augmented images:

![Augmented Image][image4]

![Augmented Image][image5]

To train the model, I will use the normalized grayscale images in the augmented dataset.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x6 					|
| Convolution 3x3	    | outputs 13x13x16								|
| RELU					|												|
| Max pooling	      	| 3x3 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | outputs 4x4x26      							|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x3x26 					|
| Fully connected		| Output = 120 									|
| RELU					|												|
| Fully connected		| Output = 84  									|
| RELU					|												|
| Fully connected		| Output = 43  									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I mainly focus on the batch size, the learning rate, and the number of epochs.

The default setting of my model is as the following:

* The batch size is 128.
* The learning rate is 0.001, falling between 0.0001 and 0.1.
* The number of epochs is 20. 

To begin with, I train the model with 3 kinds of batch sizes: 64, 128, and 256, with all the other parameters set as default. There is no much difference in model performance. Thus, I decide to use 64 in the final setting.

Then I train the model with two more learning rates: 0.0002 and 0.005. When learning rate is set as 0.005, the validation accuracy jumps up and down. When learning rate is set as 0.0002, the validation accuracy goes down steadily; however, it takes more epochs for convergence. I decide to use 0.001 as the learning rate in the final setting.  

As for the architecture of neural network, I make a few changes.

Firstly, I try using average pooling instead of max pooling. I find there is no much difference in model performance between average pooling and max pooling. I decide to use max pooling.

Secondly, I use a 3x3 filter instead of 5x5 filter for all convolution layers. Meanwhile, for the second pooling layers, I use 3x3 filter instead of 2x2 filter. In this case, more neurons and parameters are needed. I find no much difference after the adjustment. 

Then I decide to add one more convolution layer with 3x3 filter and one more pooling layer to get a deeper network. The downside of a deeper network is time-consuming. However, a deeper network does achieve better validation accuracy. I decide to use the neural network with 3 convolution layers.

Finally, I train the number of epochs by adding 10 more epochs each time and see whether early stopping is required. I find setting the number of epochs as 100 is ideal.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy   = 0.994
* validation set accuracy = 0.938
* test set accuracy of    = 0.905

![Confusion Matrix][image6]

The first architecture is chosen to be LeNet with the following setting:
* The learning rate is 0.001, which is randomly selected between 0.0001 and 0.1.
* The number of epochs is 10, which is seleted based on the computing speed of my laptop. 
* The number of batchs is 128.

LeNet is an ideal architecture because it can achieve good validation accuracy without intensive computing. I also design another deep neural network with two inception layers, two pooling layers, and 3 fully connected layers. However, the new network design is much more complicated and requires intensive computing.  

Typical adjustments made to the architecture include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

To achieve better model performance, I decide to build a deeper neural network with 3 convolution layers and 3 pooling layers. Moreover, I try to add dropout layers after the first and the second fully connection layer. However, the overfitting problem seems not severe in our case, so the effect of adding dropout layers is obscure. 

The learning rate, the batch size, and the number of epochs are tuned. Tuning the learning rate is aimed to make the model converging faster. Tuning the batch size and the number of epochs can ease the overfitting problem.  

I think VGG Net can be an ideal model design which achieve good model performance with less computing power required. VGG Net is the winner of 2014 ImageNet Competition. It already shows great performance on facial detection. The outlines of faces are similar. Likewise, there are only a few patterns of outlines of traffic signs. The architecture of my final model is similar to VGG Net. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Image 1: Pedestrians][image4] ![Image 2: Road Work][image5] ![Image 3: Slippery Road][image6] 
![Image 4: Maximum Speed Limit 120][image7] ![Image 5: Roundabout][image8]

The first image might be difficult to classify because the sign is tilt.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image				| Prediction		| 
|:------------------|:------------------| 
| Pedestrians		| Road Work  		| 
| Road Work 		| Road Work 		|
| Slippery Road		| Slippery Road		|
| 120 km/h			| 120 km/h			|
| Roundabout		| 100 km/h			|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

If the traffic sign in the image is tilt and distorted, it makes it hard to predict correctly. Based on the predictive results of the test set, it's a common mistake to predict 'Pedestrians' sign as 'Road Work' sign. However, the mistake seldom happens for the other way around. Inreasing the images of tilt 'Pedestrain' sign may improve predictive accuracy. As for 'Roundabout' sign, without grayscale transformation, this mistake may be avoided.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Please refer to 'Traffic_Sign_Classifier.ipynb' file.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Ommitted.
