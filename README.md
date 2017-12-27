# Kamil Krzyk - My Road to AI

## About

This is a repository that I have created to track my progress in AI/Data Science related topics in order to organise my knowledge and goals. Purpose of doing this is self-motivation, open source/study material for others, portfolio and TODO list.

## Table of contents
- [Kamil Krzyk - My Road to AI](#kamil-krzyk---my-road-to-ai)
- [About](#about)
- [Table of Contents](#table-of-contents)
- [AI Related Presentations](#ai-related-presentations)
- [AI Implementations](#ai-implementations)
	+ [Machine Learning](#machine-learning)
	+ [Deep Learning](#deep-learning)
	+ [Tutorials](#tutorials)
	+ [Based on Research Papers](#based-on-research-papers)
- [Algorithm Implementations](#algorithm-implementations)
	+ [Divide & Conquer](#divide--conquer)
- [Books](#books)
- [Courses & Certificates](#courses--certificates)
- [Sources]($sources)
- [Contact](#contact)

## AI Related Presentations
| Presentation  | Where | Date | Slides |
| :---: | :---: | :---: | :---: |
| Welcome to MOOC era! - My experiences with Deep Learning Foundations Nanodegree at Udacity | Speaker - GDG & Women Techmakers - Machine Learning #3 | 18.10.2017 | [Link](https://speakerdeck.com/f1sherkk/welcome-to-mooc-era-my-dlfnd-experiences-at-udacity) |
| Soft introduction into MSE based Linear Regression (part 2 of 'What this Machine Learning is all about?' talk)  | Azimo Lunch&Learn | 16.11.2017 | [Link](https://speakerdeck.com/f1sherkk/soft-introduction-to-mse-based-linear-regression) |


## AI Implementations
In this section I want to show off my knowledge about various AI related algorithms, frameworks, programming languages, libraries and more. Priority is to show how the algorithm works - not to solve complex and ambitious problems.

### Machine Learning
| Algorithm  | Description | Implementation | Dataset | Creation Date | Last Update |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | - | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/LinearRegression/raw_solution/LinearRegression_Raw.ipynb) | Generated Numbers | 18.04.2017 | 15.09.2017 |
| | - | [Python (sklearn)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/LinearRegression/sklearn_solution/LinearRegression_Sklearn.ipynb) | Generated Numbers | 18.04.2017 | 15.09.2017 |
| | - | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/LinearRegression/tensorflow_solution/LinearRegression_Tensorflow.ipynb) | Generated Numbers | 23.09.2017 | 23.09.2017 |
| Ridge Regression | Compared result with Linear Regression | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/RidgeRegression/raw_solution/RidgeRegression_Raw.ipynb) | Generated Numbers | 23.09.2017 | 23.09.2017 |
| Polynomial Regression | Approximating Polynomial of degree 2 | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/PolynomialRegression/raw_solution/PolynomialRegression_Degree2_Raw.ipynb) | Generated Numbers | 08.06.2017 | 15.09.2017 |
| | Approximating Polynomial of degree 2 | [Python (sklearn)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/PolynomialRegression/sklearn_solution/PolynomialRegression_Degree2_Sklearn.ipynb) | Generated Numbers | 10.06.2017 | 15.09.2017 |
| | Approximating Polynomial of degree 3 | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/PolynomialRegression/raw_solution/PolynomialRegression_Degree3_Raw.ipynb) | Generated Numbers | 10.06.2017 | 15.09.2017 |
| | Approximating Polynomial of degree 3 | [Python (sklearn)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/PolynomialRegression/sklearn_solution/PolynomialRegression_Degree3_Sklearn.ipynb) | Generated Numbers | 10.06.2017 | 15.09.2017 |
| Logistic Regression | Data Analysis, Kaggle Competition | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/Regression/LogisticRegression/raw_solution/LogisticRegression_Raw.ipynb) | Titanic Disaster | 19.10.2017 | 24.10.2017 |
| KNN | Manhattan, Euclidean Similarity | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/KNN/raw_solution/KNN_Iris_Raw.ipynb) | iris | 21.07.2017| 24.09.2017 |
| | Euclidean Similarity | [Python (sklearn)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/KNN/sklearn_solution/KNN_Iris_Sklearn.ipynb) | iris | 22.07.2017 | 24.09.2017 |
| PCA | - | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/PrincipalComponentAnalysis/PCA_Raw.ipynb) | Generated Numbers | 01.04.2017 | 23.09.2017 |
| K-Means Clusters | 3-dimensional data | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/K-Means/raw_solution/K-Means_VideoGames_Raw.ipynb) | Video Game Sales from Kaggle | 01.10.2017 | 05.10.2017 |
| Naive Bayes | Gaussian Distribution | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/MachineLearning/NaiveBayes/raw_solution/NaiveBayes_PimaIndiansDiabetes_raw.ipynb) | Pima Indian Diabetes | 02.11.2017 | 03.11.2017 |
| Lasso Regression | - | - | - | - | - |
| SVM | - | - | - | - | - |
| Decision Tree | - | - | - | - | - |
| Random Forest | - | - | - | - | - |

### Deep Learning

#### Multilayer Perceptron
| Problem | Description | Implementation | Dataset | Creation Date | Last Update |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Digit Classification | 2-layers, mini-batch | [Python (raw)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/MultilayerPerceptron/Classification/MNIST-Dataset/raw_solution/MultilayerPerceptron-MNIST-Raw.ipynb) | MNIST | 19.06.2017 | 14.08.2017 |
| Digit Classification | 2-layers, mini-batch, dropout-regularization | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/MultilayerPerceptron/Classification/MNIST-Dataset/tensorflow_solution/MultilayerPerceptron-MNIST-Tensorflow.ipynb) | MNIST | 29.06.2017 | 18.07.2017 |
| Digit Classification | 2-layers, mini-batch | [Python (Tensorflow + Keras)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/MultilayerPerceptron/Classification/MNIST-Dataset/tensorflow-keras_solution/MultilayerPerceptron-MNIST-TensorflowWithKerasWrapper.ipynb) | MNIST | 08.07.2017 | 18.07.2017 |
| Digit Classification | 2-layers, mini-batch | [Python (tflearn)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/MultilayerPerceptron/Classification/MNIST-Dataset/tflearn_solution/MultilayerPerceptron-MNIST-tflearn.ipynb) | MNIST | 21.06.2017 | 21.06.2017 |
| Digit Classification | 2-layers, mini-batch | [Python (Keras)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/MultilayerPerceptron/Classification/MNIST-Dataset/keras_solution/MultilayerPerceptron-MNIST-Keras.ipynb) | MNIST | 18.07.2017 | 18.07.2017 |
| Prediction of Bike Shop Clients Number | 1-layer, mini-batch | [Python (numpy, matplotlib)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/MultilayerPerceptron/Regression/Bike-Sharing-Dataset/raw_solution/MultilayerPerceptron-BikeSharing-Raw.ipynb) | Bike-Sharing | 13.08.2017 | 13.08.2017 |
| Encrypting data with Autoencoder | 1-layer Encoder, 1-layer Decoder, mini-batch | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/Autoencoder/ImageEncription/MNIST-Dataset/tensorflow_solution/MLP-Encryption-Autoencoder.ipynb) | MNIST | 13.07.2017 | 13.07.2017 |
| Detecting Text Sentiment | - | - | IMDb | - | - |

#### Convolutional Neural Net
| Problem | Description | Implementation | Dataset | Creation Date | Last Update |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Digit Classification | tf.layer module, dropout regularization, batch normalization | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/ConvNet/Classification/MNIST-Dataset/tensorflow_solution/ConvNet-MNIST-Tensorflow-BN-tflayer.ipynb) | MNIST| 16.08.2017 | 23.08.2017 |
| 10 Classes Color Images Classification | tf.nn module, dropout regularization | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/ConvNet/Classification/CIFAR-10-Dataset/tensorflow_solution/ConvNet-CIFAR10-Tensorflow-tfnn.ipynb) | CIFAR-10 | 16.08.2017 | 07.09.2017 |
| 10 Classes Color Images Classification | tf.layer module, dropout regularization | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/ConvNet/Classification/CIFAR-10-Dataset/tensorflow_solution/ConvNet-CIFAR10-Tensorflow-tflayer.ipynb) | CIFAR-10 | 16.08.2017 | 09.09.2017 |
| 10 Classes Color Images Classification | tf.layer module, dropout regularization, batch normalization | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/ConvNet/Classification/CIFAR-10-Dataset/tensorflow_solution/ConvNet-CIFAR10-Tensorflow-BN-tflayer.ipynb) | CIFAR-10 | 19.08.2017 | 10.09.2017 |

#### Recurrent Neural Network
| Problem | Description | Implementation | Dataset | Creation Date | Last Update |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Simple Language Translator | In form of my DLFND project for now | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/RNN/4thProject-LanguageTranslator/dlnd_language_translation.ipynb) | Small part of French-English corpus | 05.05.2017 | 24.05.2017 |
| "The Simpsons" Script Generation | In form of my DLFND project for now | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/RNN/3ndProject-GeneratingScriptTV/dlnd_tv_script_generation.ipynb) | "The Simpsons" script | 06.06.2017 | 14.07.2017 |

#### Generative Adversarial Neural Network
| Problem | Description | Implementation | Dataset | Creation Date | Last Update |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Generating Human Face Miniatures | DCGAN | [Python (Tensorflow)](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/DeepLearning/GAN/ImageGeneration/CELEB-Dataset/DC-GAN-FaceGeneration-Tensorflow.ipynb) | CelebA | 11.09.2017 | 13.09.2017 |

### Tutorials
Teaching others is best way of teaching yourself. I will try to create tutorials with various implementations of ML&DL models and more. Idea of my tutorials is to build models with small steps, with many comments, ideally including math and links to sources that I use to create them.

| Tutorial | Creation Date  | Last Update |
| :---: | :---: | :---: |
| [Implementing KNN with comments and basic math](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/Tutorial/KNN_Raw_Tutorial.ipynb) | 21.07.2017 | 21.07.2017 |
| [Implementing PCA with comments and basic math](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/AI-Implementations/Tutorial/PCA_Raw_Tutorial.ipynb) | 01.04.2017 | 01.04.2017 |

### Based on Research Papers
In this section I will do my best to provide implementations of models based on research papers. My target framework will be Keras or/and PyTorch.

#### Convolutional Neural Network
| Paper | Year | Implementation | Dataset | Creation Date  | Last Update |
| :---: | :---: | :---: | :---: | :---: | :---: |
| [Deep Photo Style Transfer](https://arxiv.org/pdf/1703.07511v1.pdf) | 2017 | - | - | - | - |
| [Spatial Transformer Networks - STN](https://arxiv.org/pdf/1506.02025.pdf) | 2016 | - | - | - | - |
| [You Only Look Once: Unified, Real-Time Object Detection - YOLO](https://arxiv.org/pdf/1506.02640.pdf) | 2016 | - | - | - | - |
| [Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artwork](https://arxiv.org/pdf/1603.01768.pdf) | 2016 | - | - | - | - |
| [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf) | 2016 | - | - | - | - |
| [Deep Residual Learning for Image Recognition - Microsoft-ResNet](https://arxiv.org/pdf/1512.03385.pdf) | 2015 | - | - | - | - |
| [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092v3.pdf) | 2015 | - | - | - | - |
| [A Neural Algorithm of Artistic Style - GATYS](https://arxiv.org/pdf/1508.06576.pdf) | 2015 | - | - | - | - |

#### Recurrent Neural Network
| Paper | Year | Implementation | Dataset | Creation Date  | Last Update |
| :---: | :---: | :---: | :---: | :---: | :---: |
| [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf) | 2016 |  - | - | - | - |
| [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf) | 2015 | - | - | - | - |
| [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf) | 2015 | - | - | - | - |
| [Skip-Thought Vectors](https://arxiv.org/pdf/1506.06726.pdf) | 2015 | - | - | - | - |
| [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) | 2014 | - | - | - | - |
| [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf) | 2013 | - | - | - | - |

#### Generative Adversarial Neural Network
| Paper | Year | Implementation | Dataset | Creation Date  | Last Update |
| :---: | :---: | :---: | :---: | :---: | :---: |
| [Generative Adversarial Text to Image Synthesis](https://arxiv.org/pdf/1605.05396v2.pdf) | 2016 | - | - | - | - |
| [Deep Convolutional GAN: DCGAN](https://arxiv.org/pdf/1511.06434.pdf) | 2015 | - | - | - | - |

## Algorithm Implementations
### Divide & Conquer
- [Gauss's Integer Multiplication](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/Algorithms/Gauss's%20Integer%20Multiplication.ipynb)
- [Karatsuba Multiplication](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/Algorithms/Karatsuba%20Multiplication.ipynb)
- [Inversion Counting](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/Algorithms/Counting%20Inversions.ipynb)
- [Merge Sort](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/Algorithms/Merge%20Sorting.ipynb)
- [Quick Sort](https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/Algorithms/QuickSort.ipynb)

## Books
Usually I prefer online sources for studying, but I believe in the power of books and try to fit them into my daily agenda.
### Programming related:
| Book | Author | Started | Finished |
| :---: | :---: | :---: | :---: |
| [Dive Into Python 3](http://www.diveintopython3.net/) | Mark Pilgrim | Aug 2017 | Sep 2017 |
| [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) | Al Sweigart | Sept 2017 | Oct 2017 |

### Machine Learning related:
| Book | Author | Started | Finished |
| :---: | :---: | :---: | :---: |
| [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) | Jake VanderPlas | Nov 2017 | - |

### Deep Learning related:
| Book | Author | Started | Finished |
| :---: | :---: | :---: | :---: |
| [Grokking Deep Learning](https://iamtrask.github.io/2016/08/17/grokking-deep-learning/) | Andrew Trask | Nov 2017 | - |

## Courses & Certificates
When I was younger I played a lot of computer games.  I still tend to play today a little as a form of relax and to spend time with friends that live far from me. One thing that I have very enjoyed about gaming was gathering trophies. You made an effort to complete list of challenges or get a great score and then looked at list of your achievements with satisfaction. My current self have inherited this habit and as I study on daily basis I like to gather proves that I have done something - to make it more like a game where each topic is a boss that you have to clear on hard mode. Of course what's in your head is most important but if it helps to motivate you, then why not?

- Programming languages:
	+ [Programming for Everybody (Getting Started with Python)](https://www.coursera.org/account/accomplishments/certificate/N5Y4ME8737YL?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BWd%2Fg8tA6QaSkzOZDrro4%2BA%3D%3D) (Feb 2017) (Coursera - University of Michigan - Charles Severance)
 	+ [Python Data Structures](https://www.coursera.org/account/accomplishments/certificate/H7G3GHE2WA4D?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BWd%2Fg8tA6QaSkzOZDrro4%2BA%3D%3D) (Feb 2017) (Coursera - University of Michigan - Charles Severance)
	+ [Using Python to Access Web Data](https://www.coursera.org/account/accomplishments/certificate/XU7RL8F57XZV?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BWd%2Fg8tA6QaSkzOZDrro4%2BA%3D%3D) (Feb 2017) (Coursera - University of Michigan - Charles Severance)
	+ [Using Databases with Python](https://www.coursera.org/account/accomplishments/certificate/JZCL9TVJEMAX?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BWd%2Fg8tA6QaSkzOZDrro4%2BA%3D%3D) (Feb 2017) (Coursera - University of Michigan - Charles Severance)

- Algorithms:
	+ [Divide and Conquer, Sorting and Searching, and Randomized Algorithms](https://www.coursera.org/account/accomplishments/certificate/PGA7EZJAJD6P) (Sep 2017) (Coursera - University of Stanford)

- AI related:
  + [Machine Learning](https://www.coursera.org/account/accomplishments/certificate/HF3R7P7JNS5S?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BWd%2Fg8tA6QaSkzOZDrro4%2BA%3D%3D) (Nov 2016 - Feb 2017) (Coursera - Stanford - Andrew Ng)
  + [Deep Learning Nanodegree](https://drive.google.com/file/d/0B8g_YGcjHiJBR25xRHllY1BJNXc/view?lipi=urn:li:page:d_flagship3_profile_view_base;Wd/g8tA6QaSkzOZDrro4%2BA%3D%3D) (Mar 2017 - Aug 2017) (Udacity - Siraj Raval, Mat Leonard, Brok Bucholtz + guest lessions by: Ian Goodfellow, Andrew Trask)
  + [Neural Networks and Deep Learning](https://www.coursera.org/account/accomplishments/certificate/ZQ3JAPWGB3PC) (Oct 2017) (Coursera - deeplearning.ai - Andrew Ng)
  + [Practical Machine Learning](https://drive.google.com/file/d/1L9BOkqxBEjkHuaWbhof5yD_MMA_8AZYN/view?usp=sharing) (Nov - Dec 2017)

## Sources
There is a list of sources that I have used (and found helpful in some way) or keep using in order to produce my repo content.

- Courses
	+ https://www.coursera.org/learn/machine-learning
	+ https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101
	+ https://www.coursera.org/specializations/deep-learning
	+ https://www.coursera.org/specializations/algorithms (part 1/4)
	+ https://www.coursera.org/learn/python (part 4/5)

- Online Lectures and YouTube Channels
	+ [CS231n (Winter 2016 with Andrej Karpathy)](https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
	+ [CS231n (Spring 2017)](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)
	+ [Natural Language Processing with Deep Learning (Winter 2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
	+ [Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)
	+ [Learn Tensorflow and Deep Learning without PhD](https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd)
	+ [sentdex](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)
	+ [Deep RL Bootcamp, 26-27 August 2017, Berkeley CA](https://sites.google.com/view/deep-rl-bootcamp/lectures)
	+ [Alena Kruchkova - Machine Learning Book Club](https://www.youtube.com/channel/UCF9O8Vj-FEbRDA5DcDGz-Pg/videos)

- Blogs
	+ http://kldavenport.com/
	+ https://medium.com/@karpathy
	+ https://machinelearningmastery.com/
	+ https://colah.github.io/
	+ https://distill.pub/

- Podcasts
	+ http://biznesmysli.pl/
	+ https://dataskeptic.com/podcast

- Cheatsheets
	+ https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463

- Repositories
	+ https://github.com/terryum/awesome-deep-learning-papers
	+ https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap
	+ https://github.com/fchollet/deep-learning-with-python-notebooks
	+ https://github.com/ZuzooVn/machine-learning-for-software-engineers
	+ https://github.com/eriklindernoren/ML-From-Scratch
	+ https://github.com/junyanz/CycleGAN
	+ https://github.com/llSourcell

- Other
	+ https://www.kaggle.com/
	+ https://www.tensorflow.org/
 	+ https://www.quora.com/How-can-beginners-in-machine-learning-who-have-finished-their-MOOCs-in-machine-learning-and-deep-learning-take-it-to-the-next-level-and-get-to-the-point-of-being-able-to-read-research-papers-productively-contribute-in-an-industry/answer/Andrew-Ng?share=c26bd326

## Contact
- Twitter: [@F1sherKK](https://twitter.com/F1sherKK)
- Medium: [@krzyk.kamil](https://medium.com/@krzyk.kamil)
- E-mail: krzyk.kamil@gmail.com
- Android related open-source contributions: [AzimoLabs](https://github.com/AzimoLabs)
- [LinkedIn](https://www.linkedin.com/in/kamil-krzyk-20275483/)
