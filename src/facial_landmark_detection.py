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
sys.path.insert(0, str(Path().resolve().parent.parent))
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin, IECore
import logging as logging


class FacialLandmarksDetectionModel:

    '''
    Class for defining the FaceLandmark Model Attributes.
    '''
    def __init__(self, model_name, threshold, device='CPU', extensions=None, async_mode = True, plugin=None):
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
        self.net = None
        self.net_plugin = None
        self.model_xml = model_name
        self.extensions = extensions


    def load_model(self, model_xml, cpu_extension=None):
        '''
        TODO: load the model
        '''
        self.model_xml = model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.device = device
        self.extensions = extensions

        # Initialize the plugin
        self.plugin = IECore()

        # Add anu neccessary extension
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        # deprecated in 2020 version
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)
        self.check_plugin(self.plugin)

        ## Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        ## check for unsuported layers
        unsupported_layers = [l for l in self.network.layers.keys() if l not in self.plugin.get_supported_layers(self.network)]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Please check for supported extensions.")
            exit(1)

        # Loading the Intermediate Representation (IR) IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.out_shape=self.network.outputs[self.output_blob].shape
        logging.info("Model Facial landmark Detection output shape printed : ", self.out_shape)
        return

    def predict(self, image, width, height, threshold):
        '''
        TODO: You will need to complete this method.
        This will be use to run predictions on the input image.
        '''
        # ref scource: ref source code: https://github.com/gauravshelangia/computer-pointer-controller/blob/master/src/face_detection.py
        tally = 0
        values = None
        width = image.shape[1]
        height = image.shape[0]
        img_frame = self.preprocess_input(image)
        if self.async_mode:
            self.exec_network.requests[0].async_infer(inputs={self.input_blob: img_frame})
        else:
            self.exec_network.requests[0].infer(inputs={self.input_blob: img_frame})

        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            img_frame,values = self.preprocess_output(image, outputs)
            return values, img_frame

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def preprocess_input(self, image):
        '''
        TODO: You will need to complete this method.
        Here, I define the function to preprocess the data befeoe the data into the model for inference.
        '''
        (n, c, h, w) = self.network.inputs[self.input_blob].shape
        # Reshape the image to input size of land_mark detection model
        # the input shape should be h*w = 48x48 - as described in openVino doc
        img_frame = cv2.resize(image, (w, h))
        img_frame = img_frame.transpose((2,0,1))
        img_frame = img_frame.reshape((n, c, h, w))
        return img_frame

    def preprocess_output(self, frame, outputs):
        '''
        TODO: You will need to complete this method.
        
        The reference source code:
        ##https://knowledge.udacity.com/questions/285095
        '''
        now_value = 0
        values = []
        outputs= outputs[0]
        
        xl,yl = outputs[0][0]*width,outputs[1][0]*height
        xr,yr = outputs[2][0]*width,outputs[3][0]*height

        # To draw the box for the left eye 
        xlmin = xl-20
        ylmin = yl-20
        xlmax = xl+20
        ylmax = yl+20
        
        # To draw the box for the right eye 
        xrmin = xr-20
        yrmin = yr-20
        xrmax = xr+20
        yrmax = yr+20
        
        cv2.rectangle(img_frame, (xlmin, ylmin), (xlmax, ylmax), (0, 0, 255), 2)
        cv2.rectangle(img_frame, (xrmin, yrmin), (xrmax, yrmax), (0, 0, 255), 2)
        values = [[int(xlmin),int(ylmin),int(xlmax),int(ylmax)],[int(xrmin),
                    int(yrmin),int(xrmax),int(yrmax)]]
        return img_frame, values

    def clean(self):
        """
        delete all the open instances
        :return: None
        """
        del self.plugin
        del self.network
        del self.exec_network
        del self.net
        del self.device