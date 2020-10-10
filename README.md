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
## 9.3 NLP Tweaking Models & SubWords




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

## Tokenization 
Neural networks utilize numbers as their inputs, so we need to convert our input text into numbers. 

<b> Tokenization </b> is the process of assigning numbers to our inputs, but there is more than one way to do this - should each letter have its own numerical token, each word, phrase, etc.

Tokenizing based on letters with our current neural networks doesn’t always work so well - anagrams, for instance, may be made up of the same letters but have vastly different meanings. for e.g Listen and Silent which are formed by rearragning leters. So, in our case, we’ll start by tokenizing each individual word.

With TensorFlow, this is done easily through use of a Tokenizer, found within <b> tf.keras.preprocessing.text </b>. 

If you wanted only the first 10 most common words, you could initialize it like so:
 <b> tokenizer = Tokenizer(num_words=10) </b>
 
<b> Fit on Texts </b> <br/>
Then, to fit the tokenizer to your inputs (in the below case a list of strings called sentences), you use .fit_on_texts():<br/>
 - tokenizer.fit_on_texts(sentences) 

<b> Text to Sequences </b> <br/>
From there, you can use the tokenizer to convert sentences into tokenized sequences:<br/>
 - tokenizer.texts_to_sequences(sentences)
 
<b> Out of Vocabulary Words</b> <br/>
However, new sentences may have new words that the tokenizer was not fit on. <br/>
By default, the tokenizer will just ignore these words and not include them in the tokenized sequences. <br/>
However, you can also add an “out of vocabulary”, or OOV, token to represent these words. This has to be specified when originally creating the Tokenizer object.<br/>
 - tokenizer = Tokenizer(num_words=20, oov_token=’OOV’)

<b> Viewing the Word Index </b><br/>
Lastly, if you want to see how the tokenizer has mapped numbers to words, use the tokenizer.word_index property to see this mapping.

<b> Text to Sequences </b> <br/>
Even after converting sentences to numerical values, there’s still an issue of providing equal length inputs to our neural networks - not every sentence will be the same length!<br/>

There’s two main ways you can process the input sentences to achieve this - padding the shorter sentences with zeroes, and truncating some of the longer sequences to be shorter. <br/>
In fact, you’ll likely use some combination of these.<br/>
With TensorFlow, the pad_sequences function from <b> tf.keras.preprocessing.sequence </b>  can be used for both of these tasks. <br/>
Given a list of sequences, you can specify a maxlen (where any sequences longer than that will be cut shorter), as well as whether to pad and truncate from either the beginning or ending, depending on pre or post settings for the padding and truncating arguments.<br/>
By default, padding and truncation will happen from the beginning of the sequence, so set these to post if you want it to occur at the end of the sequence.<br/>
If you wanted to pad and truncate from the beginning, you could use the following: <br/>
 - padded = pad_sequences(sequences, maxlen=10)

Steps done in Tokenization:

  - Tokenizing input text
  - Creating and padding sequences
  - Incorporating out of vocabulary words
  - Generalizing tokenization and sequence methods to real world datasets

## Word Embeddings

Embeddings are clusters of vectors in multi-dimensional space, where each vector represents a given word in those dimensions.
To create our embeddings, we’ll first use an embeddings layer, called tf.keras.layers.Embedding
It takes 3 Arguments:
  - the size of the tokenized vocabulary, 
  - the number of embedding dimensions to use, 
  - the input length (from when you standardized sequence length with padding and truncation)

The output of this layer needs to be reshaped to work with any fully-connected layers. It can be done with a pure Flatten layer, or use GlobalAveragePooling1D for a little additional computation that sometimes creates better results.
In the example shown in notebook we’re only looking at positive vs. negative sentiment, so only a single output node is needed (0 for negative, 1 for positive). We’ll be able to use a binary cross entropy loss function since the result is only binary classification.

Steps done in Embeddings:
 - transformed tokenized sequences into embeddings
 - developed a basic sentiment analysis model
 - visualized the embeddings vector
 - tweaked hyperparameters of the model to improve it
 - diagnosed potential issues with using pre-trained subword tokenizers when the network doesn’t have sequence context

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

## SubWords
Pieces of words,often made up of smaller words ,that make words with similar roots be tokenized more similarly.<br/>
This helps avoid marking very rare words as OOV when you use only the most common words in a corpus.<br/>
This can further expose an issue affecting all of our models up to this point, in that they don’t understand the full context of the sequence of words in an input.

Subword Datasets </b> </br>
There are a number of already created subwords datasets available online. If you check out the IMDB dataset on TFDS, for instance, by scrolling down you can see datasets with both 8,000 subwords as well as 32,000 subwords in a corpus (along with regular full-word datasets).</br>
We’ll use TensorFlow’s SubwordTextEncoder and its build_from_corpus function to create one from the reviews dataset we used previously.
