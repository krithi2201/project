# k-NN and K-means clustering

A program to implement k-Nearest Neighbour classification algorithm on the iris flower dataset and visualize the results.  
Demonstration of image segmentation using k-means clustering algorithm and visualizing the results.  

In statistics, the k-nearest neighbours algorithm (k-NN) is a non-parametric classification method used for classification and regression.  

k-means clustering is a method of vector quantization, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.  

k-Means Clustering is an unsupervised learning algorithm that is used for clustering whereas KNN is a supervised learning algorithm used for classification.  

The two methods are applied on the same dataset. Thus, we can differentiate between the two methods.  


# Getting started

Begin by exploring the sorce code and download the datasets. (flower1.jpeg and Iris.csv)

# Import required libraries

For k-NN  

```
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
```
For k-means clustering

```
import matplotlib.pyplot as plt
import cv2
```

# Run the python file

```
$ python3 project.py
```

# Results

For k-NN  

Train and test splits, evalution report via confusion matrix and classification, visualization of the samples using scatter plot.  


![k-NN](https://github.com/krithi2201/project/blob/main/k-NN.PNG)

For k-means clustering

Convert the data into 8-bit values, reshape data into the original image dimensions and show the concatenated image.  


![k-means](https://github.com/krithi2201/project/blob/main/k-means%20clustering.PNG)
