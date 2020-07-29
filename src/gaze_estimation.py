'''
This defines the class and attributes for the facial landmark detection model. 
'''
import numpy as np
import os
import cv2
import argparse
import time
import sys
from argparse import ArgumentParser
from pathlib import Path
import math
sys.path.insert(0, str(Path().resolve().parent.parent))
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin, IECore
import logging as logging


class GazeEstimationModel:
    '''
    Class for defining GazeEstimation Model and Attributes.
-    '''
    def __init__(self, model_name,  threshold, device='CPU', extensions=None, async_mode = True, plugin=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.out_shape = None
        self.exec_network = None
        self.threshold = threshold
        self.device = device
        self.async_mode = async_mode
        self.infer_request =None
        self.net_plugin = None
        self.net = None
        self.model_xml = model_name
        self.extensions = extensions

    def load_model(self, model_xml, gaze_angles, input_gaze_angles, cpu_extension=None):
        '''
        TODO: load models
        '''
        self.model_xml = model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.device = device
        self.extensions = extensions

        # Initializing the plugins
        self.plugin = IECore()

        # Add any neccesary extensions ##
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Reading the Intermediate Representation (IR) model as a IENetwork
        # deprecated in 2020 version
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)

        self.check_plugin(self.plugin)

        ## check for supported layer
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        ## check for unsupported layers
        unsupported_layers = [l for l in self.network.layers.keys() if l not in self.plugin.get_supported_layers(self.network)]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Please check for supported extensions.")
            exit(1)

        # Loading the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_gaze_angles = self.network.inputs['gaze_angles']
        # print(self.input_pose_angles)
        self.output_blob = next(iter(self.network.outputs))
        self.out_shape=self.network.outputs[self.output_blob].shape
        logging.info("Model Gaze Estimation output shape printed : ", self.out_shape)
        return


    def predict(self,l_eye_img,r_eye_img, target_gaze, img_frame, width, height):
        '''
        TODO: 
        The accuracy of gaze direction prediction is evaluated through the use of "mean absolute error (MAE)" of the angle
        (in degrees) between the ground truth and predicted gaze direction.
        Input_blob
        Blob in the format [BxCxHxW] where 
        B = batch size
        C = number of channels
        H = image height
        W = image width
        with the name right_eye_image and the shape[1x3x60x60]
        Blob in the format [BxC] where:
        B = batch size
        C = number of channels
        with the name head_pose_angles and the shape[1x3]

        outputs_blob
        The net outputs a blob with the shape: [1x3], containing cartesian coordinates of
        gaze direction vector. Please note that output vector is not normalized and has non-unit length.
        Output layer name in INference Engine format: gaze_vector

        Ref: https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

        '''
        ## for left and right eye image and shape
        tally = 0
        values = None
        width = l_eye_img.shape[1]
        height = l_eye_img.shape[0]
        l_eye_img,r_eye_img = self.preprocess_input(l_eye_img,r_eye_img)

        # perform inference on image shape 
        #ref: https://github.com/gauravshelangia/computer-pointer-controller/blob/master/src/facial_landmark_detection.py
        if self.async_mode:
            self.exec_network.requests[0].async_infer(inputs=
                {"gaze_angles": target_gaze,"l_eye_img":l_eye_img,
                "r_eye_img":r_eye_img})
        else:
            self.exec_network.requests[0].infer(inputs=
                {"gaze_angles": target_gaze,"l_eye_img":l_eye_img,
                "r_eye_img":r_eye_img})

        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            vout = self.preprocess_output(l_eye_img, r_eye_img, target_gaze, outputs)
            return vout

    def preprocess_input(self, l_eye_img,r_eye_img):
        '''
        TODO: You will need to complete this method.
        Here I preprocess the data before feeding the data into the model for inference.
        '''
        # left eye input shape [1,3,60,60]
        l_eye_img = cv2.resize(l_eye_img, (60, 60))
        l_eye_img = l_eye_img.transpose((2,0,1))
        l_eye_img = l_eye_img.reshape((1, 3, 60, 60))
        # and right eye input shape[1,3,60,60]
        r_eye_img = cv2.resize(r_eye_img, (60, 60))
        r_eye_img = r_eye_img.transpose((2,0,1))
        r_eye_img = r_eye_img.reshape((1, 3, 60, 60))

        return img_frame, l_eye_img,r_eye_img

    def preprocess_output(self, l_eye_img, r_eye_img, outputs, target_gaze):
        '''
        TODO: You will need to complete this method.
        Here I preprocess the model before feeding the output of this model to the next model.
        '''
        # ref source code: # Ref: https://knowledge.udacity.com/questions/254779
        gaze_vector = outputs[0]
        roll = gaze_vector[2]#pose_angles[0][2][0]
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        cs = math.cos(roll * math.pi / 180.0)
        sn = math.sin(roll * math.pi / 180.0)

        tmpX = gaze_vector[0] * cs + gaze_vector[1] * sn
        tmpY = -gaze_vector[0] * sn + gaze_vector[1] * cs

        return (tmpX,tmpY),(gaze_vector)
        # raise NotImplementedError

    def clean(self):
        """
        This function deletes all the open instances
        :return: None
        """
        del self.plugin
        del self.network
        del self.exec_network
        del self.net
        del self.device