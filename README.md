# Computer Pointer Controller

### Computer_Pointer_Controller_Using_EyeGaze

The Computer Pointer Controller is an application that is being developed to control the mouse pointer of the computer system using gaze estimation model. This application uses Inference Engine API from Intel's OpenVino Toolkit and it was tested on CPU, GPU, VPU and FPGA Devices to determine the best device suitbale for deployment. To build this application, the gaze estimation model required three inputs:

- The head pose

- The left eye image

- The right eye image


### These inputs were obtained from three different OpenVino models:

- [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)

- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)

- [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
 
## Project Pipeline

The project pipeline coordinates the flow of data from the input and amongst the different models and finally to the mouse controller. The flow of data is represented as:

![project pipleline](images/pipeline.png)


## Project Set Up and Installation

To build this project, I used the OpenVino Toolkit. Details of how to install and configure the development environment is provided here in [OpenVino](https://docs.openvinotoolkit.org/latest/index.html).

It is advisable to create and setup a virtual environment for this application. 

## Setting Up Developemnt Environment and Deocumentation

- Install `Hoomebrew` on your system using this link: https://brew.sh/

- After installing homebrew, install virtual environment: pyenv using `brew install pyenv`

- Using pyenv create a virtual environment: using the command `pyenv virtualenv <python_version> ENV_NAME`

- Note: you will have to activate the python shell using: `pyenv shell <python_version>`

- Download and Install [OpenVino](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)

- Activate the virtual environment using command line: ` pyenv activate <ENV_NAME>`

- Navigate to the folder where the `requirements.txt` is saved

- Install require binaries using `pip install -r requirements.txt`
Note: this is not an exhaustive requirment list, depending on your system, you may require to install more dependencies.

- `pip freeze` to see what's already installed and what is missing. 

## Reviewers Remarks

- modifying the `requirment.txt`f file to only include important dependencies. Please not that the user may install additional dependencies depending on the operating system, virtual environment using (i.e: pyenv, conda, virtualenv)

```
image==1.5.27
numpy==1.17.4
Pillow==6.2.1
requests==2.22.0
opencv-python
olefile==0.46
PyAutoGUI==0.9.48
PyMsgBox==1.0.7
pyobjc-core==6.2
pyobjc-framework-Cocoa==6.1
pyobjc-framework-Quartz==5.3
PyScreeze==0.1.26
PyTweening==1.0.3
pathlib
```
- To get the supported arguments, run the command: `python3 main.py -h`


**ERROR MESSAGE WHEN INSTALLING `Pyautogui`**

I solved this problem using the following command: 

- git clone https://github.com/asweigart/pyautogui
- cd pyautogui
- sudo python3 setup.py install

- Ref source code:  https://stackoverflow.com/questions/34939986/how-to-install-pyautogui

### Directory Structure
The project directory contains the bin, images, models and src folders:

- bin folder: containing the video 
	
- images folder coantains the project pipeline 

- model frolder: conatains the pretrained IR models
	
- src folder that contains the application codes

## Reviewers Remarks

- Adding project directory structure as recommended by the reviewer

```

.
├── bin
│   └──  demo.mp4
├── images/                                    # contains the project pipeline/ suported images for README.md
├── models/                                    # contains model files
├── README.md
├── requirements.txt
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── main.py
    ├── model.py
    ├── mouse_controller.py
```

## Downloading the models

- `python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 --output_dir models`
- `python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output_dir models`
- `python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --output_dir models`
- `python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output_dir models`

**NOTE:** You might have to install yaml using the command `pip install pyyaml` to enable downloading the models.

## Running the application 

The application can be run with the following command:

- Using video file as input:
`$ python3 src/main.py -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hp models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -fl models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -ge models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -o . -d "CPU" -t 0.5 -m 'async' -so 'yes'`

- Using webcam as input:

`$ python3 src/main.py -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hp models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -fl models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -ge models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i 'cam' -o . -d "CPU" -t 0.5 -m 'async' -so 'yes'`


## Stand Out Suggestions
There is challenge encountered when installing the `pyautogui`, which was encountered due to missing requiremnts. Therefore it is advisable to make a comprehensive list of the dependencies to allow easy installation of `pyautogui`.

Another impotatant suggestion is that `pyautogui` doesn not support python versions <3.6, therefore it is important to set up a virtual environment with python versions > 3.6 for the project.

**NOTE:** I encountered this error when attempting to execute the application: `RuntimeWarning: Compiletime version 3.6 module openvino.inference.ie.pi` does not match runtime version 3.7.

This problem can be encountered when trying to run your application in python 3.7 virtual environment while your openvino python path is version 3.6. So you can mitigate this by creating a virtual environment with python version 3.6 or change your openvino pythin path in bash to match the python version in your virtual environment. 

In my case my openvino python version was set to 3.6.5 so to run the application I change to a python shell with version 3.6 using the `pyenv` command: `pyenv shell 3.6.5`

### Async Inference
This application applies `async inference` when executing the models. This is because `async inference` have the ability to perform multiple inference at the same time. This approach allows the application to save time and energy required by the device compared to the `sync inference` method that takes more processing time and conusmes more time and energy. 

### Edge Cases
In this project, I demonstrated the ability of using human gaze to control the computer pointer. Howeever, there can be certain limitations associated with this application. One such limitation is that the application can only work better when only one person is detected in the video frame. For example in real setting when using the web cam, the application can be programmed to address others detected in the video frame. One major approach could be that we programme the application to detect only the person with the main focus and occupy the largest surface area in the screen. Another pressumptive problem is one that is associated with the screen size. It is important to calibrate the screen size, such that the application can adapt to every screen ratio to make the cursor movement more precise.

**References:**
All studied materials and research are fully refernce:

- https://knowledge.udacity.com/questions/171017

- https://knowledge.udacity.com/questions/254779

- https://knowledge.udacity.com/questions/257795

- https://knowledge.udacity.com/questions/242851

- https://knowledge.udacity.com/questions/285095

- https://github.com/gauravshelangia/computer-pointer-controller/blob/master/src/face_detection.py
- https://knowledge.udacity.com/questions/242566

- https://stackoverflow.com/questions/34939986/how-to-install-pyautogui


