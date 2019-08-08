<p align="center">
<img src="https://www.inaccel.com/wp-content/uploads/logo-horizontal1200px.png" width=60% height=60% align="middle" alt="InAccel"/>
</p>

# Gaussian Naive Bayes Classifier

This is an FPGA accelerated solution of Gaussian NaiveBayes classification algorithm. It provides up to **700x** speedup compared to a single threaded execution on an Intel Xeon CPU.

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
|      2018.2 and above     |

## Design Files

-   The application code is located in the *hosts_srcs* directory. Accelerator kernel files are located under the *kernel_srcs* directory while any accelerator binaries will be compiled to the current directory.
-   The *Makefile* will help you generate any host executable and accelerator _.xclbin_ files.

A list of all the project files is shown below:

	- data/
    - hosts_srcs/
    	- NaiveBayes.cpp
    - kernel_srcs/
    	- Classifier_0.cpp (Accelerated kernel)
    	- Classifier_1.cpp (Accelerated kernel)
    	- Classifier_2.cpp (Accelerated kernel)
    	- Classifier_3.cpp (Accelerated kernel)
	- Makefile
	- sdaccel.ini

## Compiling the kernels
This step is **optional** for running the demo as you can use a **pre-compiled** version of NaiveBayes Classifier accelerator found in our bitstream repository.

To compile the kernels you just need to execute `make xbin`.  
A full list of all the available Makefile targets can be found using `make help` command.

As far as the **platform** (or board) is concerned, Makefile uses **AWS_PLATFORM** environment variable as the target platform for the kernels compilation. If you are running this on AWS make sure AWS_PLATFORM environment variable is present and points to the platform DSA files<sup>1</sup>. Otherwise you can set Makefile `PLATFORM` variable to point to your platform DSA files.

1.  To obtain the AWS platform DSA files make sure you have cloned aws-fpga github repository and followed the setup instructions.

## Running the Demo

To run the demo you first have to download the dataset upon which we will evaluate our model. Then you are going to setup the environment for compiling the host application.

* **Download Necessary Datasets:**  
Download letters dataset to **data** directory
	``` bash
	wget https://s3.amazonaws.com/inaccel-demo/data/nist/letters_csv_train.dat -O data/letters_csv_train.dat
	```

* **Setup Inaccel and Coral API**  
<p align="center">
<img src="https://www.inaccel.com/wp-content/uploads/coral_logo_big-1-e1561553344239.png" width=60% height=60% align="middle" alt="InAccel Coral"/>
</p>

The host code sends requests for acceleration to Coral FPGA Resource Manager through the Coral API. To use Coral please follow the instructions below.  You can find **full documentation** in [InAccel Docs](https://docs.inaccel.com/latest/).

1. [Install **InAccel CLI**](https://docs.inaccel.com/latest/inaccel/install/rpm/).
1. [Setup your Environment.](https://docs.inaccel.com/latest/tutorial/setup/)
1. [Install **Coral API**.](https://docs.inaccel.com/latest/tutorial/api/c++/)
1. **Install the bitstreams you are going to use with Coral**.
	* To install the **pre-compiled** bitstream for NaiveBayes Classification found in our [bitstream repository](https://store.inaccel.com/artifactory/webapp/#/artifacts/browse/tree/General/bitstreams/com/inaccel/ml/NaiveBayes/1.0/xilinx/aws-vu9p-f1-04261818/dynamic_5.0/Classifier) simply execute the following:
		``` bash
		inaccel install https://store.inaccel.com/artifactory/bitstreams/com/inaccel/ml/NaiveBayes/1.0/xilinx/aws-vu9p-f1-04261818/dynamic_5.0/Classifier
		```
	* To install your **newly compiled** bitstream do the following:
		1. Create a folder. Download into this folder the bitstream.json file from our [bitstream repository](https://store.inaccel.com/artifactory/webapp/#/artifacts/browse/tree/General/bitstreams/com/inaccel/ml/NaiveBayes/1.0/xilinx/aws-vu9p-f1-04261818/dynamic_5.0/Classifier/bitstream.json), that fully describes the design of the generated bitstream (number and names of kernels, number of each kernel's arguments etc.)
			``` bash
			wget https://store.inaccel.com/artifactory/bitstreams/com/inaccel/ml/NaiveBayes/1.0/xilinx/aws-vu9p-f1-04261818/dynamic_5.0/Classifier/bitstream.json
			```
		1. Copy the generated xcblin file (after you have completed the Amazon AFI creation procedure) to the folder you created. Make sure that the name of the copied file is **NaiveBayes.xclbin**
		1. Install the bitstream to the Coral FPGA Manager. Give the (relative or absolute) path of the folder you created in the inaccel install command.  
			``` bash
			inaccel install <path/to/folder_name>
			```
1. Execute ```inaccel list``` to make sure that the bitstream is installed successfully.
1. **Start InAccel Coral**:  
Due to a bug in XRT 2018.3 version, we recommend that you also mount **sdaccel.ini** file found in the project root directory when starting inaccel to avoid any errors. Navigate to the project folder and execute the following:
	```bash
	inaccel stop && inaccel start --only -v ${PWD}/sdaccel.ini:/opt/inaccel/jre/bin/sdaccel.ini
	```
1. **Compile the demo application:**  
For the CPU only version execute ```make``` while for the FPGA-accelerated one execute ```make host```.
1. **Run the demo application:**  
	```bash
	./NaiveBayes
	```
