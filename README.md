## 1. Celsius to Fahrenheit using tensoflow
## 2. Image Classification using Fashion MNIST dataset using tensorflow
## 3. Predict Fuel Efficiency(Miles Per Gallon) using tensorflow
## 4. Fashion MNIST ImageClassification Using CNN
## 5. DogsVsCats without Augmentation using CNN
## 6. DogsVsCats with Augmentation using CNN
## 7. Flower Classification with Augmentation using CNN
## 8. Loading and Saving Models
## 9.1 NLP Word Tokenization 
## 9.2 NLP Word Embeddings





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

## Transfer Learning
We can use <b> Transfer Learning</b> to create very powerful Convolutional Neural Networks with very little effort. The main key points of this lesson are:

 - Transfer Learning: A technique that reuses a model that was created by machine learning experts and that has already been trained on a large dataset. When performing transfer learning we must always change the last layer of the pre-trained model so that it has the same number of classes that we have in the dataset we are working with.
 - Freezing Parameters: Setting the variables of a pre-trained model to non-trainable. By freezing the parameters, we will ensure that only the variables of the last classification layer get trained, while the variables from the other layers of the pre-trained model are kept the same.
 - MobileNet: A state-of-the-art convolutional neural network developed by Google that uses a very efficient neural network architecture that minimizes the amount of memory and computational resources needed, while maintaining a high level of accuracy. MobileNet is ideal for mobile devices that have limited memory and computational resources.
 
You also used transfer learning to create a Convolutional Neural Network that uses MobileNet to classify images of Dogs and Cats. You were able to see that transfer learning greatly improves the accuracy achieved in the Dogs and Cats dataset. 

<b>Note:</b> fit_generator is used in cases where batches are coming from a generator (ImageDataGenerator) instead of fit.

## Word Embeddings

Embeddings are clusters of vectors in multi-dimensional space, where each vector represents a given word in those dimensions.
To create our embeddings, we’ll first use an embeddings layer, called tf.keras.layers.Embedding
It takes 3 Arguments:
  - the size of the tokenized vocabulary, 
  - the number of embedding dimensions to use, 
  - the input length (from when you standardized sequence length with padding and truncation)

The output of this layer needs to be reshaped to work with any fully-connected layers. It can be done with a pure Flatten layer, or use GlobalAveragePooling1D for a little additional computation that sometimes creates better results.
In the example shown in notebook we’re only looking at positive vs. negative sentiment, so only a single output node is needed (0 for negative, 1 for positive). We’ll be able to use a binary cross entropy loss function since the result is only binary classification.


## Some of the potential things you might tweak to better predict sentiment from text.
Data and preprocessing-based approaches
  - More data
  - Adjusting vocabulary size (make sure to consider the overall size of the corpus!)
  - Adjusting sequence length (more or less padding or truncation)
  - Whether to pad or truncate pre or post (usually less of an effect than the others)
  
Model-based approaches
  - Adjust the number of embedding dimensions
  - Changing use of Flatten vs. GlobalAveragePooling1D
  - Considering other layers like Dropout
  - Adjusting the number of nodes in intermediate fully-connected layers

