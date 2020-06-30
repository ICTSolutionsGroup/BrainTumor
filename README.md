# BrainTumor
Brain Tumor Classification Using Deep Learning
Ahmad Saleh1, Rozana Sukaik2 and Samy S. Abu-Naser3

Brain tumor is very common and destructive malignant tumor diseases that leads to short life if it is not diagnose early enough. Brain tumor classification is very critical step after detection of the tumor to able to attain an effective treatment plan. The research paper aims to increase the level and efficiency of MRI machines in classifying brain tumors and identifying their types, Using AI Algorithm, CNN and Deep Learning.  We have trained our brain tumor dataset using five pre-retrained models: Xception, ResNet50, InceptionV3, VGG16, and MobileNet. The f-score accuracy of un seen images was 98.75%, 98.50%, 98.00%, 97.50%, and 97.25% respectively. These accuracies have a positive impact on early detection of tumors before the tumor causes Physical side effects, such as paralysis and others.

There is a very large group of people, whose exact numbers are unknown but they continue to increase, who are diagnosed with a type of brain tumor called secondary brain tumor. Early detection is always likely to accelerate the process of controlling and eliminating the tremor at early stages, with the help of highly efficient clinical imaging devices. Meanwhile, patients who suffer from brain tumors face the problem of MRI machines’ s inability to precisely detect and classify the brain tumor, which could lead to physical complications that cause disability[1 ].
Types of tumors: 
●        glioma tumor .
●        meningioma tumor.
●        no tumor .
●        pituitary tumor .

This research paper discusses the problems caused by the inability of MRI machines in identifying and classifying tumors.  These tumors could cause complications such as physical disabilities, which would then force patients to seek proper rehabilitation in order to treat or reduce the disabilities. Moreover, the complications of brain tumors on brain’s functionality varies depending on the location, size and type of these tumors. A patient may become unable to move because a tumor could put pressure on the area that controls the body's movement in the brain. It could also cause loss of sight or hearing [ 1-8 ].
II. Background
Deep Learning :
Deep learning allows one to build multiple processing layers to teach machine representations of data with multiple levels. These methods improved the state-of-the-art in speech recognition, object detection and many other domains. Learning can be supervised or unsupervised . In deep learning, The Input data at all levels is converted into a more abstract and structured representation. In an Tumor recognition application, the raw input will represent to a matrix of pixels, the first representational layer may abstract the pixels and encode edges of tumor, the second layer may compose and encode arrangements of edges of tumor, the third layer may encode a representation of circle, and the fourth layer may recognize that the image contains tumor. Importantly, a deep learning process can learn which features to optimally place in which level on its own[ 2].
 Supervised learning : 
“Supervised learning is a learning model built to make prediction, given an unforeseen input instance. A supervised learning algorithm takes a known set of input dataset and its known responses to the data (output) to learn the regression/classification model. A learning algorithm then trains a model to generate a prediction for the response to new data or the test dataset “. an open loop chain. In this case, the end-effector is related to the base through one kinematic chain. In the case of parallel manipulators, the links are connected in a manner that forms a closed loop chain. The end-effector is located at the end of the chain, but is related to the base through two, or several kinematic chains. Serial manipulators have, in general, the advantage of more flexible and wider working space compared to parallel manipulators.
Unsupervised Learning : 
“Unsupervised learning is the training of an Artificial Intelligence (AI) algorithm using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance. An AI system may group unsorted information according to similarities and differences even though there are no categories provided. AI systems capable of unsupervised learning are often associated with generative learning models, although they may also use a retrieval-based approach (which is most often associated with supervised learning). Chatbots, self-driving cars, facial recognition programs, expert systems and robots are among the systems that may use either supervised or unsupervised learning approaches”[ 4]
CNN : 
In neural networks, Convolutional neural network (ConvNets or CNNs) is one of the main categories to do images recognition, images classifications. Objects detections, recognition faces etc., are some of the areas where CNNs are widely used, sometimes researchers uses matlab, in out project we are using Python Libraries (tensorflow, pandas, openCv, Keras ..etc)[5-12 ]. 

Datasets : 
The dataset Imported from Kaggle [6 ] has 2880 images for training and validation. Each brain tumor type has 520 images for training and 200 images for validation. 800 (each type has 200 images) un-seen test images were used to test the final trained models. All images are 256*256 pixels in size. The number of images on meningioma tumor dataset is   not enough for training. Thus, we used augmentation and  resizing for the images to i abunaser@alazhar.edu.ps3, increase the number of images and overcome the problem of overfitting during training due to the limited number of images.
Pituitary Brain tumors : 
Pituitary Brain tumors are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in too many of the hormones that regulate important functions of your body. Some pituitary tumors can cause your pituitary gland to produce lower levels of hormones. Most pituitary tumors are noncancerous (benign) growths (adenomas). Adenomas remain in your pituitary gland or surrounding tissues and don't spread to other parts of your body. Pituitary tumors are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in too many of the hormones that regulate important functions of your body. Some pituitary tumors can cause your pituitary gland to produce lower levels of hormones. Most pituitary tumors are noncancerous (benign) growths (adenomas). Adenomas remain in your pituitary gland or surrounding tissues and don't spread to other parts of your body[8-10  ].

Glioma Brain Tumor : 
Glioma is a type of tumor that occurs in the brain and spinal cord. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function. Three types of glial cells can produce tumors. Gliomas are classified according to the type of glial cell involved in the tumor, as well as the tumor's genetic features, which can help predict how the tumor will behave over time and the treatments most likely to work. Signs and symptoms of a Glioma tumor : Headache, Nausea or vomiting, Confusion or a decline in brain function, Memory loss, Personality changes or irritability, Difficulty with balance, Urinary incontinence, Vision problems, such as blurred vision, double vision or loss of peripheral vision, Speech difficulties and Seizures, especially in someone without a history of seizures [ 7 ].
             Meningioma Brain Tumor : 
      A meningioma is a tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Although not technically a brain tumor, it is included in this category because it may compress or squeeze the adjacent brain, nerves and vessels. Meningioma is the most common type of tumor that forms in the head. Signs and symptoms of a meningioma typically begin gradually and may be very subtle at first. Depending on where in the brain or, rarely, spine the tumor is situated, signs and symptoms may include: Changes in vision, such as seeing double or blurriness Headaches, especially those that are worse in the morning Hearing loss or ringing in the ears Memory loss Loss of smell Seizures Weakness in your arms or legs Language difficulty [9-10  ].


Xception : 
Xception is a deep convolutional neural network architecture, which developed by google that involves depthwise separable convolutions. and it developed to present an Interpretation of Inception Modules in CNN.
Images first go through the entry flow, then through the middle flow which is repeated eight times, and finally through the exit flow[15 ].

Figure 3: Xception Algorithm
      Deep Learning Model : 
     ResNet-101 Residual learning can be easily interpreted as subtraction of input features learned from that layer. This is done by ResNet using shortcut connections to each pair of 33 ﬁlters, directly connecting the input of kth layer to (k + x)th layer. The motive behind bypassing layers is to keep away the problem of vanishing gradients by reutilizing activations from the preceding layer till the layer next to the present one has learned its weights[14 ]. 
While training the network, weights will amplify the layer next to the present one and will also adjust to mute the preceding layer. ResNet-101 is 101-layer Residual Network and is a modiﬁed version of the 50-layer ResNet.

Figure(4) - Block Diagram of Residual Module and SE-ResNet module .
   Inception-v3 :
      Convolutional neural network architecture, Inception v3, used in this study. Inception v3 network stacks 11 inception modules where each module consists of pooling layers and convolutional filters with rectified linear units as activation function. The input of the model is two-dimensional images of 16 horizontal sections of the brain placed on 4 3 4 grids as produced by the preprocessing step. Three fully connected layers of size 512, 512, and 3 are added to the final concatenation layer. A dropout with rate of 0.6 is applied before the fully connected layers as means of regularization. The model is pretrained on ImageNet dataset and further fine-tuned with a batch size of 8 and learning rate of 0.0236.
Figure 5: Inception-v3
       MobileNet :
         MobileNet is a CNN architecture model for Image Classification and Mobile Vision.There are other models as well but what makes MobileNet special that it very less computation power to run or apply transfer learning to.This makes it a perfect fit for Mobile devices,embedded systems and computers without GPU or low computational efficiency with compromising significantly with the accuracy of the results.It is also best suited for web browsers as browsers have limitation over computation,graphic processing and storage.
MobileNets for mobile and embedded vision applications is proposed, which are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. The core layer of MobileNet is depthwise separable filters, named as Depthwise Separable Convolution. The network structure is another factor to boost the performance. Finally, the width and resolution can be tuned to trade off between latency and accuracy[ 17].
III. Related Works
As for the methods that have been used to solve this problem in the past, it would be mostly relying on doctors’ and specialists’ consultations, where the ratio of errors varies from one to another. Thus, we will be making major upgrades and additions to the MRI and clinical examination systems using Deep Learning, and Convolutional Neural Networks (CNN) . The classification of images is based on the principle of taking an image as an entry, classifying the entry into a specific category of four, and at a high resolution that exceeds 95%, which would then lead to having efficient imaging devices and accurate, fast and early detection and classification of tumors. There are many models designed to classify Brain Tumors using Deep Convolutional Neural Network (CNN). But there is no model for the classification of specialized types of these tumors or to improve efficiency of MRI machines[ 18 ] .
IV. Results
We have trained validated the five pre-trained models for brain tumor classification: Xception, ResNet50, InceptionV3, VGG16, and MobileNet. The f-score accuracies and error loss are shown in table 1 and in figure . We have tested the five models with un seen images and the accuracies were 98.75%, 98.50%, 98.00%, 97.50%, and 97.25% respectively (as seen in Table 1). These accuracies have a positive impact on early detection of tumors before the tumor causes Physical side effects, such as paralysis.
Table 1: results of the five models
Algorithm
Training
f-score Accuracy
Validation
f-score Accuracy
Training
f-score Loss
Validation
f-score Loss
Testing
Accuracy
Xception
100.00%
97.04%
0.0004
0.1381
98.75%
ResNet50
99.50%
96.76%
0.0291
0.2006
98.50%
InceptionV3
99.29%
95.12%
0.0236
0.2281
98.00%
VGG16
100.00%
0.9478
0.00004
0.3316
97.50%
MobileNet
100.00%
92.35%
0.0014
0.2829
97.25%


The research paper and the project aim to increase the level and efficiency of MRI machines in classifying tumors and identifying their types, and here the main goal was achieved, which is classification of brain tumors and the accuracy of testing and training was with Accuracy not less than 97.25, which has a positive impact on early detection of tumors before the tumor causes effects Physical side, such as paralysis and others .
	
