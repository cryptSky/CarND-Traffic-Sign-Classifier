# **Traffic Sign Recognition** 
 
 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[hist_1]: ./train_hist.png "Training set histogram"
[hist_2]: ./valid_hist.png "Validation set histogram"
[hist_3]: ./test_hist.png "Test set histogram"
[normal]: ./normal.png "Normal image"
[grayscale]: ./grayscale.png "Greyscale image"
[final]: ./final.png "Final image before feeding it into neural net"

[01]: ./01.png "Traffic Sign 1"
[02]: ./02.png "Traffic Sign 2"
[03]: ./03.png "Traffic Sign 3"
[04]: ./04.png "Traffic Sign 4"
[05]: ./05.png "Traffic Sign 5"

Link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the dataset

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data classes are distributed across training, validation and test set

![alt text][hist_1]
![alt text][hist_2]
![alt text][hist_3]

### Design and Test a Model Architecture

#### 1. Preprocessing image data

As a first step, I decided to convert the images to grayscale because color information is not really important. Note that, there is no two traffic signs which are different only by color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][normal]
![alt text][grayscale]

As a last step, I normalized the image data because it will be better for neural net to deal with normalized pixel values thus better train the model.

Here is an example of a final image before feeding it into neural net:
![alt text][final]

#### 2. Model architecture

I decided to use use model similar to LaNet but with slightly modified number of feture maps and neurons in hidden layers
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32   				|
| Fully connected, 800  | outputs 400 									|
| RELU					|												|
| Fully connected, 400  | outputs 200 									|
| RELU					|												|
| Fully connected, 200  | outputs 43 - number of classes				|
| Softmax				|           									|

#### 3. Training the model

To train the model, I used an average softmax cross entropy as the cost function and Adam optimizer on top of it.
Learning rate  - 0.001
Batch size - 256
Number of epochs - 400

#### 4. Model and approach

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.950
* test set accuracy of 0.940

I chose LaNet architecture as the base architecture. I think it is a good one for traffic sign recognition because it is not very large, so it will perform inference fast and has good accuray on similar tasks. Firstly, I used it without any changes, it was a bit challenging to get 0.93 accuracy on validation set using that architecture. I played with such parameters as number of epochs, learning rate and batch size to improve accuracy of the model. Finally, I was able to do that by setting lerning rate to 0.001, 400 epochs and setting batch size to 256. I left those parameters untouched after changing the architecture and they perform well too. So after looking into accuracy you can see that this model performs really well.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][01] ![alt text][02] ![alt text][03] 
![alt text][04] ![alt text][05]

They have better look comparing to other images from the dataset, so my model should perform well on them.

#### 2. Model's predictions on new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)	| Speed limit (60km/h)							| 
| No passing     		| No passing									|
| Road work			    | Road work			    						|
| Children crossing  	| Children crossing				 				|
| Pedestrians			| Pedestrians       							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.94

#### 3. Predictions

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

For all images my model performs well giving 4 images approx. 1.000000000e+00 softmax probability, so I won't include here detailed table of those. 

Will include only the top five soft max probabilities of "Road work" sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.93350208e-01		| Road work         	 				    	| 
| 6.64980058e-03		| Bumpy road        					    	|
| 2.70198108e-09		| Road narrows on the right					    |
| 1.97108654e-15	    | Slippery road              			    	|
| 1.04187915e-17		| General caution                   	    	|


#### 1. Output of trained network's feature maps
Visualizations of the feature maps(in IPython notebook) show what kind of featres model uses to determine traffic sign type.




