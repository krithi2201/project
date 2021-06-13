
# A program to implement k-Nearest Neighbour classification algorithm on the iris flower dataset and #visualize the results.  
import numpy as np
from collections import Counter
#Euclidean Distance
def  euclidean_distance(x1,  x2): return np.sqrt(np.sum((x1-x2)**2))
class k_nearest_neighbors: def 	init__(self, k):
self.k = k
def knn_fit(self, X_train, y_train):
self.X_train = X_train self.y_train = y_train
def knn_predict(self, X):
predicted_labels = [self._predict(x) for x in X] return np.array(predicted_labels)

#helper method
def _predict(self,x):
#compute distances
distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
#get k nearest samples, labels
k_indices = np.argsort(distances)[:self.k] k_nearest_labels = [self.y_train[i] for i in k_indices]
 
#majority vote, most common class label
majority_vote =  Counter(k_nearest_labels).most_common(1) return majority_vote[0][0]

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix import numpy as np
import matplotlib.pyplot as plt

#Loading dataset
iris_data = datasets.load_iris() data = iris_data.data
target = iris_data.target
#Train/Test splits
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2) print("training instance: ",len(X_train))
print("test instance: ",len(X_test))
#Train KNN model
my_model = k_nearest_neighbors(k = 3) model=my_model.knn_fit(X_train, y_train) predictions = my_model.knn_predict(X_test)
#Evaluation report
print("confusion  Matrix:") print(confusion_matrix(y_test,predictions))             print("Classification report:", classification_report(y_test, predictions))
#Visualize the predictions
for class_value in range(3):
row_ix = np.where(predictions== class_value)
 
row_px = np.where(y_test== class_value)
#create scatter of these samples
if(class_value==0): m='+'
c='red'
elif(class_value==1): m="o"
c='green' elif(class_value==2):
m='x' c='yellow'

plot1 = plt.figure(1)
plt.plot(X_test[row_ix, 1], X_test[row_ix, 0],marker=m,color=c)
#create scatter of these samples
if(class_value==0): m='+'
c='violet' elif(class_value==1):
m="o" c='black'
elif(class_value==2): m='x'
c='cyan'

plot2 = plt.figure(2)
plt.plot(X_test[row_ix, 1], X_test[row_px, 0],marker=m,color=c) plt.show()

#Demonstration of image segmentation using k-means clustering algorithm and visualizing the results.
import numpy as np
import matplotlib.pyplot as plt import cv2

#Read in the image
image = cv2.imread('flower1.jpeg')

#Change color to RGB (from BGR)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) plt.imshow(image)
#Reshaping the image into a 2D array of pixels and 3 color values(RGB) pixel_vals = image.reshape((-1,3))

#Convert to float type
pixel_vals = np.float32(pixel_vals)

#The below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are the run or the epsilon(which is the required accuracy) #becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

#Then perform k-means clustering with h number of clusters defined as 3 #also random centres are initially chosen for k-means clustering
k=2
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#Convert data into a 8-bit values centers = np.uint8(centers)
 
segmented_data = centers[labels.flatten()] print(centers)
print(segmented_data)

#Reshape data into the original image dimensions segmented_image = segmented_data.reshape((image.shape)) plt.imshow(segmented_image)

image_concat = np.concatenate((image, segmented_image), axis=1) plt.imshow(image_concat)
