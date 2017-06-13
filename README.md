#UCLA CS 249 Final Project
Intel & MobileODT Cervical Cancer Screening

##Deployment
This project is based on TFLearn library and requires GPU instance to run it. We will show you how to deploy the enviroment on Amazon Web Service ec2 instance.

###EC2 GPU Instance
For machine image (AMI), we choose "Ubuntu Server 16.04 LTS (HVM)". In this example we choose p2.xlarge instance. Next, add at least 60 GB storage for your instance and follow the AWS instruction to configure and start the instance.

###Installing the CUDA Toolkit and cuDNN
Please follow the instructions on this [page](http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/)

###Tensorflow Installation
Tensorflow is the prerequisite to work with TFLearn. Please follow the official documentation to install it. We will show you the steps based on our configurations.
First, export the binary to install
```
# Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl
```
Then install TensorFlow
```
# Python 3
$ sudo pip3 install $TF_BINARY_URL
```

###TFLearn Installation
For the latest stable version:
```
pip install tflearn
```
For more details please get reference from the official website.
