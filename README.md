# 1. Celsius to Fahrenheit using tensoflow
# 2. Image Classification using Fashion MNIST dataset using tensorflow
# 3. Predict Fuel Efficiency(Miles Per Gallon) using tensorflow
# 4. Fashion MNIST ImageClassification Using CNN
# 5. DogsVsCats without Augmentation using CNN
# 6. DogsVsCats with Augmentation using CNN
# 7. Flower Classification with Augmentation using CNN






## Techniques to Prevent Overfitting

 1. Early Stopping: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.
 2.Image Augmentation: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.
 3. Dropout: Removing a random selection of a fixed number of neurons in a neural network during training.


## CNNs with RGB Images of Different Sizes:

Resizing: When working with images of different sizes, you must resize all the images to the same size so that they can be fed into a CNN.

Color Images: Computers interpret color images as 3D arrays.

RGB Image: Color image composed of 3 color channels: Red, Green, and Blue.

Convolutions: When working with RGB images we convolve each color channel with its own convolutional filter. Convolutions on each color channel are performed in the same way as with grayscale images, i.e. by performing element-wise multiplication of the convolutional filter (kernel) and a section of the input array. The result of each convolution is added up together with a bias value to get the convoluted output.

Max Pooling: When working with RGB images we perform max pooling on each color channel using the same window size and stride. Max pooling on each color channel is performed in the same way as with grayscale images, i.e. by selecting the max value in each window.

Validation Set: We use a validation set to check how the model is doing during the training phase. Validation sets can be used to perform Early Stopping to prevent overfitting and can also be used to help us compare different models and choose the best one.
