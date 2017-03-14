## Vehicle Detection and Tracking

### Using a linear SVM classifier on various color and gradient features, this project implements a pipeline that detects and tracks vehicles on highways in a dashcam video footage.

------

**Vehicle Detection Project**

The goals / steps of this project are the following:

- Extract the following features on a labeled training set of images:
  - Binned spatial color features
  - Histogram of colors
  - Histogram of Oriented Gradients (HOG)
- Train a classifier Linear SVM classifier
- Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
- Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Optimize the heat map with a moving average to reduce the number or false-positives
- Estimate a bounding box for vehicles detected.

[//]: #	"Image References"
[image1]: ./output_images/carsnotcars.png
[image2]: ./output_images/search.png
[image3]: ./output_images/0.png
[image4]: ./output_images/1.png
[image5]: ./output_images/2.png
[image6]: ./output_images/3.png
[image7]: ./output_images/4.png
[image8]: ./output_images/5.png
[video1]: ./processed_video.mp4



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The first step is to read in all the vehicle and non-vehicle images. In the third code block of the `Vehicle-Tracking.ipynb` notebook, the 64x64 images form GTI, KITTI and other vehicle image libraries are loaded recursively into two arrays of file name strings: `cars` and `not_cars`. A simple length check verifies that both classes have similar number of examples, which is important as an unbalanced training set can create bias in the classifier:

```N
Number of Cars:  12180
Number of Not Cars:  10784
```

Next, in code block #4, one random image from each class is chosen and has its feature vector extracted with the `single_img_features` function, defined in the helper functions code block #2. This function takes various parameters and computes the final vector by concatenating the binned spatial features, histogram of colors, and HOGs features in that order.

For the HOGs features, the `skimage.hog()` API was used to produce both the feature vector and the visualization. Using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, the following visualizations were obtained for the two random images:

![alt text][image1]

Intuitively, one can notice that the gradients have larger magnitudes on the edges of the rectangular shape of the car, around the rear windshield, and around the license plate. At the same time, the non-car images do not have recognizable patterns in its HOGs. This difference will contribute to the high accuracy of the classifier. 

#### 2. Explain how you settled on your final choice of HOG parameters.

The optimal parameters were found in code block #5, by training a classifier and trying to increase both the prediction accuracy as well as the bounding box accuracy on the test images. In order to iterate quickly, the training for parameter optimization was performed using a random sample of 1000 images from each of the two classes. The labeled data was only separated into training and test sets without the extra validation set, because the video to be processed was considered a test set and wasn't used until all parameters were tweaked.

The following parameters produced satisfactory results on both the classification accuracy and the bounding boxes:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656]
```

Most of the parameter tweaking only involved the color space, HOG channel, spatial size and histogram bins.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Again in code block #5, the `sklearn.svm.LinearSVC` classifier was trained using the labeled data. This classifier was chosen because of its power of classifying large feature vectors into few final classes without taking too much processing time. Because the feature vector is simply a result of concatenation of different features, it is important to scale it before passing it to the classifier. This was done using `sklearn.preprocessing.StandardScaler`. With the chosen parameters, the classifier took 43.23 seconds to train 9744 images, each with a feature vector of length 8460. The test accuracy reached 0.9996. Although this accuracy turns out to be good enough, it is important to keep in mind that a slightly lower accuracy here may produce a lot of false positives as hundreds of window searches are done on each image.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The `slide_window` function was originally used to find all the windows in the image given the windows size, the region of interest, and the overlap percentages. It goes through all the pixels in the ROI with each step's size equal to the number of overlapped pixels. Then, the `search_windows` function would extract the features for each window, and use the classifier to predict whether this is a "car" window that needs to be kept. The size of the window was chosen to be 64x64, the same size as the training images to simplify the pipeline and avoid unnecessary computation. The overlapping percentage was chosen to be 75% to strike a good balance between search accuracy and search time: searching with a low window overlap might miss some cars while searching with a high overlap wastes resources and increases processing time. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Using the methods described above, the following image was obtained:

![alt text][image2]

It does a good job of producing bounding boxes only around the cars for most of test images. A more detailed analysis will follow in subsequent sections.

One problem with searching all the windows first then extract features one-by-one is that the expensive HOG features extraction is done hundreds of times for a single image. This quickly becomes a problem for video processing where ideally we would like to keep the processing speed above 10 fps so the human eye will not notice the difference.

In order to improve the efficiency, the `find_cars` function was implemented so the hog features are extracted only once per frame and classifier prediction is done at the same time as windows searching.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:



### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./processed_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

From the left column of the images below, it is easy to notice the occasional false-positive detections where a non-car object is detected as a car. In order to filter these out, a heat map was implemented. For every detected bounding box, the pixels inside of it get a +1 in heat. Since multiple overlapping bounding boxes are expected around cars, a threshold can be applied to reset the low heat parts of the map to zero, assuming that the detection algorithm only produced a few isolated false-positives. Then `scipy.ndimage.measurements.label()` is used to identify contiguous non-zero heat regions, which are then bounded by single boxes, as shown in the right column.  

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

While the pipeline already produces relatively good results at this point, the final labeled bounding boxes often jump around from frame to frame, and produce false negatives in some cases. In order to alleviate this problem, a moving average of heat maps was implemented in `find_cars` function. A queue of the 10  latest heat maps is maintained. With every new frame, a new heat map is produced and replaces the oldest heat map in the queue. Then, all heat maps in the queue are averaged and then thresholded. This produces a much smoother final bounding box, and helps with false-negatives because the frames with missed detections are averaged out by previous and subsequent frames with positive detections.



------

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One challenge of this project was the large number of parameters that could affect the end result. Since the combinations grow exponentially, it has not been easy to quickly find the optimal values for all of them. 

A weakness of the current implementation is that the bounding boxes' sizes sometimes change quickly from frame to frame because the temporal continuity of the box size is not taken into account. A similar approach as the heat maps' moving average can be used on the box size to solve this problem.

Another potential improvement for the pipeline is to use the knowledge of the perspective transform, especially when this project is combined with the advanced lane finding project where the transform is already computed, to search for vehicles with various windows sizes. This will at the same time save search time and improve accuracy near the bottom and side edges where vehicles are expected to be larger. 