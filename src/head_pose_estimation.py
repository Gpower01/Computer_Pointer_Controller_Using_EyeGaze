'''
This defines the class and attributes for the face detection model. 
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



class HeadPoseEstimationModel:

    '''
    Class for defining the HeadPoseEstimation Model Attributes.
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
        self.net_plugin = None
        self.net = None
        self.extensions = extensions
        self.model_xml = model_name


    def load_model(self, model_xml, cpu_extension=None):
        '''
        TODO: load models
        '''
        self.model_xml = model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.device = device
        self.extensions = extensions

        # Initializing the plugins
        self.plugin = IECore()

        # Add a CPU extension and any neccessary extension
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Reading  the Intermediate Representation (IR) model as a IENetwork
        # IENetwork deprecated in 2020 version
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)
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
        logging.info("Model Head Pose Detection output shape printed : ", self.out_shape)
        return

    def predict(self, image, width, height, threshold):
        '''
        TODO: You will need to complete this method.
        To run predictions on the input image.
        '''
        # [1,3,60,60]
        tally = 0
        valuess = None
        width = image.shape[1]
        height = image.shape[0]
        img_frame = self.preprocess_input(image)
        if self.async_mode:
            self.exec_network.requests[0].async_infer(inputs={self.input_blob: img_frame})
        else:
            self.exec_network.requests[0].infer(inputs={self.input_blob: img_frame})
        
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs
            person_in_frame, target_gaze = self.preprocess_output(image, outputs)
            return person_in_frame, target_gaze

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape


    def preprocess_input(self, image):
        '''
        TODO: You will need to complete this method.
        preprocessing the input shape
        '''
        # [1,3,60,60]
        (n, c, h, w) = self.network.inputs[self.input_blob].shape
        img_frame = cv2.resize(image, (w, h))
        img_frame = img_frame.transpose((2,0,1))
        img_frame = img_frame.reshape((n, c, h, w))
        return img_frame

    def preprocess_output(self, image, outputs, width, height):
        '''
        TODO: You will need to complete this method.
        Output layer names in Inference Engine format:
        name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        Each output contains one float value  (yaw, pitсh, roll).
        '''
        # To Parse head pose detection results
        # ref: source code: https://knowledge.udacity.com/questions/242566
        pitch = outputs["angle_p_fc"][0] 
        yaw = outputs["angle_y_fc"][0]
        roll = outputs["angle_r_fc"][0]
        # Draw output
        if ((yaw > -22.5) & (yaw < 22.5) & (pitch > -22.5) &
                (pitch < 22.5)):
            return True,[[yaw,pitch,roll]]
        else:
            return False,[[0,0,0]]

    # code source: https://knowledge.udacity.com/questions/171017
    def draw_axes(self, img_frame, center_of_face, yaw, pitch, roll):
        focal_length = 950.0
        scale = 50

        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])
        # R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # R = np.dot(Rz, np.dot(Ry, Rx))
        R = Rz @ Ry @ Rx
        # print(R)
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(img_frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(img_frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(img_frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(img_frame, p2, 3, (255, 0, 0), 2)
        return img_frame

    # code source: https://knowledge.udacity.com/questions/171017
    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix

    def clean(self):
        """
        deletes all the open instances
        :return: None
        """
        del self.plugin
        del self.network
        del self.exec_network
        del self.net
        del self.device