'''
This defines the class and attributes for the face detection model. 
'''
import socket
import os
import sys
import json
import time
import cv2
from collections import namedtuple
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from threading import Thread
import logging as logging
from face_detection import FaceDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from facial_landmark_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from mouse_controller import MouseController
from input_feeder import InputFeeder


def args_parser():
    """
    Parse command line arguments.

    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--fdmodel", required=True, type=str,
                        help="Path to an .xml file with a trained model"
                        "face detection model")
    parser.add_argument("-hp", "--hpmodel", required=True,
                        help="Path to an .xml file with a trained model"
                        "head pose estimation model")
    parser.add_argument("-fl", "--flmodel", required=True,
                        help="Path to an .xml file with a trained model"
                        "facial landmarks detection model")
    parser.add_argument("-ge", "--gemodel", required=True,
                        help="Path to an .xml file with a trained model"
                        "gaze estimation model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image."
                        "'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                        "path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                        "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-t", "--threshold", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument("-o", "--out_dir", help = "Path to output directory", type = str, default = None)
    parser.add_argument("-m", "--mode", help = "async or sync mode", type = str, default = 'async')
    parser.add_argument("-so", "--save_output", default=None, type=str,
                        help="Select between yes | no ")

    return parser

POSE_CHECKED = False


def main():
    """
    Initialise the inference network, stream video to network
    and output stats and video
    :param args: Command line arguments parsed by build_argsparser()
    :return: None
    """
    # mouse movement ("low", "medium", "fast")
    global POSE_CHECKED
    mouse_movement = MouseController("low", "fast")

    logging.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=logging.INFO, stream=sys.stdout)
    args = args_parser().parse_args()
    logging_message = logging.getLogger()

    if args.input == 'cam':
       input_feed = 0
    else:
        input_feed = args.input
        assert os.path.isfile(args.input), "Missing files or Specified input file doesn't exist or entered correctly"

    # Ref: source code: https://stackoverflow.com/questions/33834708/cant-write-video-by-opencv-in-python/33836463
    # Ref: source code: https://knowledge.udacity.com/questions/275173
    cap = cv2.VideoCapture(input_feed)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    vout = cv2.VideoWriter(os.path.join(args.out_dir, "vout.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height), True)
    
    if args.save_output == 'yes':
        vout_fd = cv2.VideoWriter(os.path.join(args.out_dir, "vout_fd.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height), True)
        vout_fl = cv2.VideoWriter(os.path.join(args.out_dir, "vout_fl.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height), True)
        vout_hp = cv2.VideoWriter(os.path.join(args.out_dir, "vout_hp.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height), True)
        vout_ge = cv2.VideoWriter(os.path.join(args.out_dir, "vout_g.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height), True)
    
    box_count = 0

    working = 1

    infer_time_start = time.time()

    if input_feed:
        cap.open(args.input)
        # Adjust delays to match the number of Frame Per Seconds in the video file

    if not cap.isOpened():
        logging_message.error("ERROR MESSAGE! Corrupt video file")
        return

    if args.mode == 'sync':
        async_mode = False
    else:
        async_mode = True

    # Initialising the class variables
    # ref: https://github.com/gauravshelangia/computer-pointer-controller/blob/master/src/main.py
    if args.cpu_extension:
        fd_model = FaceDetectionModel(args.fdmodel, args.threshold,extensions=args.cpu_extension, async_mode = async_mode)
        hp_model = HeadPoseEstimationModel(args.hpmodel, args.threshold,extensions=args.cpu_extension, async_mode = async_mode)
        fl_model = FaceLandmarksDetectionModel(args.flmodel, args.threshold,extensions=args.cpu_extension, async_mode = async_mode)
        ge_model = GazeEstimationModel(args.gemodel, args.threshold,extensions=args.cpu_extension, async_mode = async_mode)
    else:
        fd_model = FaceDetectionModel(args.fdmodel, args.threshold, async_mode = async_mode)
        hp_model = HeadPoseEstimationModel(args.hpmodel, args.threshold, async_mode = async_mode)
        fl_model = FaceLandmarksDetectionModel(args.flmodel, args.threshold, async_mode = async_mode)
        ge_model = GazeEstimationModel(args.gemodel, args.threshold, async_mode = async_mode)

    # Load the model through ##
    # And infer network
    logging_message.info("================ Models loading time ======================")
    start_time = time.time()
    fd_model.load_model()
    logging_message.info("Face Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    hp_model.load_model()
    logging_message.info("Headpose Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    fl_model.load_model()
    logging_message.info("Facial Landmarks Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )


    start_time = time.time()
    ge_model.load_model()
    logging_message.info("Gaze Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )
    logging_message.info("========================== End ============================")

    model_load_time = time.time() - infer_time_start

    logging.info("All models are loaded successfully")

    while cap.isOpened():
        flag, img_frame = cap.read()
        if not flag:
            print ("checkpoint *UNRECORDED")
            break

        box_count += 1
        gazing = 0
        POSE_CHECKED = False

        if img_frame is None:
            logging.error("checkpoint ERROR! EMPTY FRAME")
            break

        width = int(cap.get(3))
        height = int(cap.get(4))

        # Asynchronous Request
        inf_start_fd = time.time()
        
        # Display the results of the output layer of the model network
        # ref source code: https://knowledge.udacity.com/questions/285095
        values, img_frame = fd_model.predict(img_frame)
        
        if args.save_output == 'yes':
            vout_fd.write(img_frame)

        fd_dur_time = time.time() - inf_start_fd
        
        if len(values) > 0:
            [xmin,ymin,xmax,ymax] = values[0] 
            head_is_moving = img_frame[ymin:ymax, xmin:xmax]
            inf_start_hp = time.time()
            person_in_frame, target_gaze = hp_model.predict(head_is_moving)
            if args.save_output == 'yes':
                p = "Target Gaze {}, Person in Frame? {}".format(target_gaze,person_in_frame)
                cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 255), 2)
                vout_hp.write(img_frame)

            if person_in_frame:
                hp_dur_time = time.time() - inf_start_hp
                POSE_CHECKED = True
                inf_start_fl = time.time()
                values,marking = fl_model.predict(head_is_moving)
                
                img_frame[ymin:ymax, xmin:xmax] = marking
                
                if args.save_output == "yes":
                    vout_fl.write(img_frame)

                fl_dur_time = time.time() - inf_start_fl
                [[xlmin,ylmin,xlmax,ylmax],[xrmin,yrmin,xrmax,yrmax]] = values
                l_eye_img = marking[ylmin:ylmax, xlmin:xlmax]
                r_eye_img = marking[yrmin:yrmax, xrmin:xrmax]

                output,gaze_vector = ge_model.predict(l_eye_img,r_eye_img,target_gaze)
                #ref: source code: https://knowledge.udacity.com/questions/264973
                if args.save_output == 'yes':
                    p = "Gaze Vector {}".format(gaze_vector)
                    cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (255, 0, 0), 1)
                    left_frame = draw_gaze(l_eye_img, gaze_vector)
                    right_frame = draw_gaze(r_eye_img, gaze_vector)
                    marking[ylmin:ylmax, xlmin:xlmax] = left_frame
                    marking[yrmin:yrmax, xrmin:xrmax] = right_frame
                    # cv2.arrowedLine(f, (xlmin, ylmin), (xrmin, yrmin), (0,0,255), 5)
                    vout_ge.write(img_frame)

                if box_count%10 == 0:
                    mouse_movement.move(output[0],output[1])
        # Drawing and documenting performance stat
        # ref: https://github.com/gauravshelangia/computer-pointer-controller/blob/master/src/main.py
        # ref source code: https://knowledge.udacity.com/questions/257795
        inf_time_message = "Face Detection Inference time: {:.3f} ms.".format(fd_dur_time * 1000)
        #
        if POSE_CHECKED:
            cv2.putText(frame, "Head Pose Estimation Inference time: {:.3f} ms.".format(hp_dur_time * 1000), (0, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img_frame, inf_time_message, (0, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 255), 2)
        vout.write(img_frame)
        if box_count%10 == 0:
            print("Inference time = ", int(time.time()-infer_time_start))
            print('Box count {} and duration {}'.format( box_count, duration))
        if args.out_dir:
            final_infer_time = time.time() - infer_time_start
            with open(os.path.join(args.out_dir, 'stats.txt'), 'w') as marking:
                marking.write(str(round(final_infer_time, 1))+'\n')
                marking.write(str(box_count)+'\n')

    if args.out_dir:
        with open(os.path.join(args.out_dir, 'stats.txt'), 'a') as marking:
            marking.write(str(round(model_load_time))+'\n')

    # Clean all models
    fd_model.clean()
    hp_model.clean()
    fl_model.clean()
    ge_model.clean()
    # release cv2 cap
    cap.release()
    cv2.destroyAllWindows()
    # release all resulting ouputs writer
    vout.release()
    if args.save_output == 'yes':
        vout_fd.release()
        vout_hp.release()
        vout_fl.release()
        vout_ge.release()

#ref: source code: https://github.com/gauravshelangia/computer-pointer-controller/blob/master/src/main.py
def draw_gaze(screen_img, gaze_pts, gaze_colors=None, scale=4, return_img=False, cross_size=16, thickness=10):

    """ This function draws an "x"-shaped cross on a screen for given gaze points while ignoring the missing ones
    """
    width = int(cross_size * scale)
   
    draw_cross(screen_img, gaze_pts[0] * scale, gaze_pts[1] * scale, 
        (0, 0, 255), width, thickness)
    return  screen_img

def draw_cross(bgr_img,x, y,color=(0, 255, 0), width=2, thickness=0.5):

    """ This function draws an "x"-shaped cross at (x,y)
    """
    x, y, w = int(x), int(y), int(width / 2)  

    cv2.line(bgr_img, (x - w , y - w), (x + w , y + w), color, thickness)
    cv2.line(bgr_img, (x - w , y + w), (x + w, y - w), color, thickness)

if __name__ == '__main__':
    main()
    sys.exit()