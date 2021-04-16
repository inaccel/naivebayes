<p align="center">
<img src="https://www.inaccel.com/wp-content/uploads/logo-horizontal1200px.png" width=60% height=60% align="middle" alt="InAccel"/>
</p>

# Gaussian Naive Bayes Classifier

This is an FPGA accelerated solution of Gaussian NaiveBayes classification algorithm. It provides up to **100x** speedup compared to a single threaded execution on an Intel Xeon CPU.

## Specifications

|  Classes |  Features  |
| :------: | :--------: |
| up to 64 | up to 2048 |

## Supported Platforms and XRT

|            Board            |
| :-------------------------: |
|      [Xilinx Alveo U200](https://www.xilinx.com/products/boards-and-kits/alveo/u200.html)      |
|      [Xilinx Alveo U250](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html)      |
|   [AWS VU9P (F1 instances)](https://aws.amazon.com/ec2/instance-types/f1/)   |
| Alibaba VU9P (F3 instances) |
| Any other Xilinx platform with at least the same amount of VU9P resources |

|            XRT            |
| :-----------------------: |
|      2019.1 and above     |

## Design Files

-  The accelerator kernel files are located under the *cpp/kernel_srcs* directory while any accelerator binaries will be compiled under cpp one.
-   The *Makefile* under cpp folder will help you generate the host executable ass well as the accelerator _.xclbin_ file.

A list of all the project files is shown below:

	- data/
	- cpp/
		- hosts_srcs/
			- NaiveBayes.cpp
		- kernel_srcs/
			- Classifier_0.cpp (Accelerated kernel)
			- Classifier_1.cpp (Accelerated kernel)
			- Classifier_2.cpp (Accelerated kernel)
			- Classifier_3.cpp (Accelerated kernel)
		- Makefile
		- sdaccel.ini
	- java/
		- src/
			- main/java/com/inaccel/ml/NaiveBayes.java
			- test/java/NaiveBayesTest.java
	- python/
		- \__init__.py
		- NaiveBayes.py
		- NaiveBayesTest.py
	- hosts_srcs/
		- NaiveBayes.cpp

## Compiling the kernels
This step is **optional** for running the demo as you can use a **pre-compiled** version of NaiveBayes Classifier accelerator found in our [bitstream repository](https://store.inaccel.com/artifactory/webapp/#/artifacts/browse/tree/General/bitstreams).

To compile the kernels you just need to execute `make xbin`.  
A full list of all the available Makefile targets can be found using `make help` command.

As far as the **platform** (or board) is concerned, Makefile uses **AWS_PLATFORM** environment variable as the target platform for the kernels compilation. If you are running this on AWS make sure AWS_PLATFORM environment variable is present and points to the platform DSA files<sup>1</sup>. Otherwise you can set Makefile `PLATFORM` variable to point to your platform DSA files.

1.  To obtain the AWS platform DSA files make sure you have cloned aws-fpga github repository and followed the setup instructions.

## Running the Demo

To run the demo you first have to download the dataset upon which we will evaluate our model. Then you are going to setup the environment for compiling the host application for the C++ case. For the Java implementation we use maven to build the project so maven needs to be installed in your system. By default the jars will be generated under `$HOME/jars` directory. You can change that by modifying the  provided **pom.xml** file. For the python case you will need to have installed python version 3 as well as the python3 package installer.

* **Download Necessary Datasets:**  
Create a folder called **data** under your home directory and download there MNIST letters dataset
	``` bash
	mkidr ~/data
	wget https://s3.amazonaws.com/inaccel-demo/data/nist/letters_csv_train.dat -O ~/data/letters_csv_train.dat
	```

* **Setup Inaccel and Coral API**  
<p align="center">
<img src="https://www.inaccel.com/wp-content/uploads/coral_logo_big-1-e1561553344239.png" width=60% height=60% align="middle" alt="InAccel Coral"/>
</p>

The host code sends requests for acceleration to Coral FPGA Resource Manager through the Coral API. To use Coral please follow the instructions below.  You can find **full documentation** in [InAccel Docs](https://docs.inaccel.com/).

1. [Install **InAccel CLI**](https://docs.inaccel.com/install/rpm/).
1. [Setup your Environment](https://docs.inaccel.com/get-started/part2/).
1. [Install **Coral API**](https://setup.inaccel.com/coral-api/?cpp).
1. Install **Coral API** for **python**:
	``` bash
	pip3 install coral-api
	```
1. **Install the bitstreams you are going to use with Coral**.
	* To install the **pre-compiled** bitstream for NaiveBayes Classification found in our [bitstream repository](https://store.inaccel.com/artifactory/bitstreams/xilinx/aws-vu9p-f1/dynamic_5.0/com/inaccel/ml/NaiveBayes/1.0/4Classifier) simply execute the following:
		``` bash
		inaccel install https://store.inaccel.com/artifactory/bitstreams/xilinx/aws-vu9p-f1/dynamic_5.0/com/inaccel/ml/NaiveBayes/1.0/4Classifier
		```
	* To install your **newly compiled** bitstream do the following:
		1. Create a folder. Download into this folder the bitstream.json file from our [bitstream repository](https://store.inaccel.com/artifactory/bitstreams/xilinx/aws-vu9p-f1/dynamic_5.0/com/inaccel/ml/NaiveBayes/1.0/4Classifier/bitstream.json), that fully describes the design of the generated bitstream (number and names of kernels, number of each kernel's arguments etc.)
			``` bash
			wget https://store.inaccel.com/artifactory/bitstreams/xilinx/aws-vu9p-f1/dynamic_5.0/com/inaccel/ml/NaiveBayes/1.0/4Classifier/bitstream.json
			```
		1. Copy the generated xcblin file (after you have completed the Amazon AFI creation procedure) to the folder you created. Make sure that the name of the copied file is **NaiveBayes.xclbin**
		1. Install the bitstream to the Coral FPGA Manager. Give the (relative or absolute) path of the folder you created in the inaccel install command.  
			``` bash
			inaccel install <path/to/folder_name>
			```
1. Execute ```inaccel list``` to make sure that the bitstream is installed successfully.
1. **Start InAccel Coral**:  
	```bash
	inaccel start
	```
1. **Compile the demo application:**  
For the C++ CPU only version execute ```make``` while for the FPGA-accelerated one execute ```make host```.
For the Java version use maven as described above, while for python use python3 executable.
1. **Run the demo application:**  
	For the C++ implementation the executable takes 2 arguments as input. The number of threads to execute the classification on software and whether you want to run classification on CPU or FPGA.
	```bash
	./NaiveBayes 8 1
	```
	For the Java implementation the command is the following. It adds all required classes to classpath and invokes java binary with NaiveBayesTest as the main class.
	```bash
	classpath=''; \
	for jar in `ls ${HOME}/jars/*.jar`; do classpath+=:${jar}; done; \
	java -cp ${classpath} NaiveBayesTest
	```
	And for the Python implementation:
	```bash
	python3 NaiveBayesTest
	```
