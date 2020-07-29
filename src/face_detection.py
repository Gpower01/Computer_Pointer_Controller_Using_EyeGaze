'''
This defines the class and attributes for the face detection model. 
'''
import os
import cv2
import time
import sys
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin, IECore
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
import argparse
from argparse import ArgumentParser
import logging as logging



class FaceDetectionModel:
    '''
    Class for Face Detection Model and attributes.
    '''
    def __int__(self, model_name, threshold, device='CPU', extensions=None, async_mode = True, plugin=None):
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
        self.infer_request = None
        self.net.plugin = None
        self.net = None
        self.model_name = model_name
        self.extensions = extensions

    def load_model(self, model_xml, cpu_extension=None):
        '''
        TO load models
        '''
        self.model_xml = model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.device = device
        self.extensions = extensions
        
        # Initialize the plugin
        self.plugin = IECore()
        self.plugin = IEPlugin(device=self.device)

        #self.plugin = IEPlugin(device=self.device)
        # Add a CPU extension, and any neccessary extension

        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Reading the Intermediate Representation (IR) model as a IENetwork
        # IENetwork deprecated in 2020 version
        self.network = self.plugin.read_network(self.model_xml, weights=model_bin)
        self.check_plugin(self.plugin)

        ## Check for supported layers
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
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.out_shape=self.network.outputs[self.output_blob].shape
        logging.info("Model Detection output shape printed : ", self.out_shape)
        return


    def predict(self, image, width, height, threshold):
        '''
        TODO: You will need to complete this method.
        This method will be use for running predictions on the input image.
        ref source code: https://github.com/gauravshelangia/computer-pointer-controller/blob/master/src/face_detection.py
        '''
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
        Before feeding the data into the model for inference,
        you might have to preprocess tehe model and this function will allow you to do that.
        '''
        (n, c, h, w) = self.network.inputs[self.input_blob].shape
        img_frame = cv2.resize(image, (w, h))
        img_frame = img_frame.transpose((2,0,1))
        img_frame = img_frame.reshape((n, c, h, w))
        return img_frame

    def preprocess_output(self, img_frame, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # [1, 1, N, 7]
        # [image_id, label, conf, xmin, ymin, xmax, ymax]
        now_value = 0
        values = []
        for box in outputs[0][0]:
            # Draw a bounding box/boxes
            confidence = box[2]
            if confidence > float(threshold):
                if box[3] < 0:
                    box[3] = -box[3]
                if obj[4] < 0:
                    box[4] = -box[4]
                xmin = int(box[3] * width) - 10
                ymin = int(box[4] * height) - 10
                xmax = int(box[5] * width) + 10
                ymax = int(box[6] * height) + 10
                # Draw the box on image
                cv2.rectangle(img_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                now_value = now_value + 1
                values.append([xmin,ymin,xmax,ymax])
                break 
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