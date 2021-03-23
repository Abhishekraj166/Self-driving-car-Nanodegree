# **Traffic Sign Classification using Convolutional Neural Networks + TensorFlow** 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration


I used the NumPy library to calculate summary statistics of the traffic
signs data set:

- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of each sign in the 
training and testing sets.

![](assets/distribution.png)

~ Exploratory Data Analysis
==========================

- After examining the provided training dataset for the distribution of the various classes, the classes were found to be highly imbalanced indicating the need to generate new data for the under-represented classes such as class-19, class-27, class-37 to name a few.

- The images differ significantly in terms of contrast and brightness. As a result, we will need to apply some kind of data processing to improve feature extraction.

### Design and Test a Model Architecture

* For Data Pre-processing, I decided to apply [Contrast Limited Adaptive Histogram Equalization](http://www.cs.utah.edu/~sujin/courses/reports/cs6640/project2/clahe.html) algorithm in the [Lab color space](https://en.wikipedia.org/wiki/Lab_color_space) to do the contrast enhancement.

* In addition to contrast enhancement, I normalized the images between 0-1.

* The following functions from [TFLearn Library](http://tflearn.org/data_augmentation/#image-augmentation) were used for **image augmentation**. The reason behind the image augmentation is the fact that our original training set was highly imbalanced. 

As stated above, our original training data was highly imbalance. Feeding our ConvNet with a balanced dataset helps it learn better.
I've decided to generate augmented data through three techniques by adding the following trasnformations.
- Rotation (-10 , 10 )
- Shifting (-5 , 5 )
- Shearing (-5 , 5 )

Then for each image, I generated 5 additional images with 5 by randomly rotating, shifting and shearing it.

After Image Augmentation, our training set size was (208794, 32, 32, 3) as it's  shown as the output of the 20th code cell in IPython notebook.

The following figure is the new distribution of signs for each class.

![](assets/distribution1.png)

- Max observations: 12060
- Min observations: 1080


Here is an example of a traffic sign image afte data augmentation

![](assets/example.png)



Here is the same example after data pre processing.

![](assets/example1.png)

The provided data had the data splitted into training and validation sets.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU	                |                                               |
| Dropout  				| 60%											    |
| Convolution	      	| 1x1 stride,  VALID padding, outputs 10x10x16 	|			|
| RELU          	    |      							                |
| Dropout       		| 60%      								     	|
| Flatten				|        								     	|
| Fully Connected       | Input = 1600    Output = 400                  |      						|												|
| RELU					|												|
| Fully Connected       | Input = 400   Output = 120                    |
| RELU                  |                                               |
| Fully Connected(logits)| Input = 120 Output = 43                      |             |


To train, validate, and test the model, I used 

- Number of training examples = 208794 
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

My final model results were:
* training set accuracy of **99.9 %**
* validation set accuracy of **96.4%**
* test set accuracy of **95.4%**

My model is based on Yann Lecun's LeNet Architecture. I added two dropout layers to prevent the ConvNet from co-adapting too
much which made overfitting less likely.

### Testing a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text](test-signs/image1.jpg)

![alt text](test-signs/image2.jpg)

![alt text](test-signs/image3.jpg)

![alt text](test-signs/image4.jpg)

![alt text](test-signs/image5.jpg)

The fourth image might be difficult to classify due to low brightness and contrast.
The fifth image might also be difficult to classify because it's rotated...

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

![](assets/predicted.png)

1. Image-1 is identified **(correctly)**
2. Image-2 is identified **(correctly)**
3. Image-3 is identified **(correctly)**
4. Image-4 is identified **(incorrectly)**
5. Image-5 is identified **(correctly)**

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set.

The code for making predictions on my final model is located in the 71th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is an 'Ahead Only' sign (probability of 0.999), and the image does contain an Ahead Only sign. 

For the second image, the model is relatively sure that this is a 'Bumpy Road' sign (probability of 0.997), and the image does contain a Bumpy Road sign.

For the third image, the model is relatively sure that this is a 'General Caution' sign (probability of 0.999), and the image does contain General Caution sign.

For the fourth image, the model is relatively sure that this is a 'Speed limit(50km/h)' sign (probability of 0.99). Though this prediction is incorrect, the image contains a Speed limit sign. It's clear that the model was not very confident about this one. 

For the fifth image, the model is relatively sure that this is a 'Stop' sign (probability of 0.999), and the image does contain a Stop sign.

The following is the bar chart of the top five soft max probabilities for each predicted sign:
![](assets/probabilities.png)
 





