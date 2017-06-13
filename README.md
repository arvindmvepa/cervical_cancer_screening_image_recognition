# UCLA CS 249 Final Project
Kaggle Competition: [Intel & MobileODT Cervical Cancer Screening](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening)

## Deployment
This project is based on TFLearn library and requires GPU instance to run it. We will show you how to deploy the enviroment on Amazon Web Service ec2 instance.

### EC2 GPU Instance
For machine image (AMI), we choose "Ubuntu Server 16.04 LTS (HVM)". In this example we will use p2.xlarge instance. Next, add at least 60 GB storage for your instance and follow the AWS instruction to configure and start the instance.

### Installing the CUDA Toolkit and cuDNN
Download [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), choose Linux OS, x86_64 Architecture, and Ubuntu distribution, the version is 16.04. Then follow the [official documentation](http://docs.nvidia.com/cuda/index.html#installation-guides) to install.
To install cuDNN, a dveloper account is required. You can download after signing up.
After downloading, you need to add following lines to your `~/.bashrc` file
```
# CUDA Toolkit
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}
```
then `source ~/.bashrc`
One can also follow the instructions on this [page](http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/).

### Tensorflow Installation
Tensorflow is the prerequisite to work with TFLearn. Please follow the official documentation to install it. We will show you the steps based on our configurations.
First, export the binary path
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

### TFLearn Installation
For the latest stable version:
```
pip install tflearn
```
For more details please get reference from the official website.

To test if you have everything working, you can try
```
git clone https://github.com/tflearn/tflearn.git
python3 tflearn/tutorials/intro/quickstart.py
```

## Download Data
For the image data, we have uploaded to dropbox, you can download the dataset by using `wget`
```
wget https://www.dropbox.com/s/l5iftr4dt9dm9pm/test.7z
wget https://www.dropbox.com/s/939zawe9d33de76/train.7z
```

You can also download the dataset from [Kaggle competition page](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data) directly.

After uncompressing the test and train set, there are couples of thing need to do.
There is a corrupted image in train set, you need to delete it.
```
rm Type_1/1339.jpg
```

You may get warning message like `UserWarning: Possibly corrupt EXIF data.`, if you wish you can use `jhead` to remove EXIF data
```
jhead -de *.jpg
```

## Running the experiment
If you wish to utilize the HDF5 format loading, you can also download `h5py` package. It will improve the performance significantly.

Now you can start to run the scripts. You need to change the dataset path to your local setting, and uncomment any dataset building statement if necessary.
The filename indicates the corresponding experiment. For example, `googlenet.py` is the original experiment we made for the GoogLeNet architecture; `googlenet_exp1.py` is the first round hyperparameter tuning test.
For more details and what each experiment does, please refer to our final report.

## Team
- Arvind Vepa
- Hao Wu
- Xin Wang
- Ziye Xing

