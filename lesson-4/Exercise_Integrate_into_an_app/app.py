import argparse
import cv2
from inference import Network
import numpy as np
INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
CONFIDENCE_THRESHOLD = 0.5
DEFAULT_COLOR = (0,0,0)
def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    t_desc = "Confidence Threshold to draw boundig boxes"
    c_desc = "Bounding Boxes Color"
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-t", help=t_desc, default=CONFIDENCE_THRESHOLD)
    optional.add_argument("-c", help=c_desc, default=DEFAULT_COLOR)

    
    args = parser.parse_args()
    return args
def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    net = Network()
    ### TODO: Load the network model into the IE
    net.load_model(model=args.m,device=args.d,cpu_extension=CPU_EXTENSION)
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the frame
        input_shape = net.get_input_shape()
        input_h = int(input_shape[2])
        input_w = int(input_shape[3])
        dims=(input_h, input_w)
        pp_frame = cv2.resize(frame,dims)
        #print(dims, pp_frame.shape)
        pp_frame = np.swapaxes(pp_frame,-1,0)
        pp_frame = np.expand_dims(pp_frame,0)
        ### TODO: Perform inference on the frame
        net.async_inference(pp_frame)
        ### TODO: Get the output of inference
        output = net.extract_output()
        
        if args.t:
            threshold = float(args.t)
            if threshold >= 1 or threshold <= 0:
                print('Please insert a confidence value in the open range (0,1)')
                exit(1)
        else:
            threshold = 0.5
            
        if args.c!=DEFAULT_COLOR:
            bgr = []
            try:
                for value in args.c.split(','):
                    value = float(value)
                    bgr.append(value)
                bgr = list(bgr)
                assert len(bgr)==3,'Please, insert only 3 values in BGR color code'
            except:
                print('Please insert comma separated values bewteen 0 and 255 in BGR color code (no space)')
                exit(1)
        else:
            bgr = (0,0,0)
        
        for bb in output[0]:
            for label in bb:
                if sum(label)>0:
                    conf = label[2]
                    if conf > threshold:
                        xmin = int(label[3]*frame.shape[0])
                        ymin = int(label[4]*frame.shape[1])
                        xmax = int(label[5]*frame.shape[0])
                        ymax = int(label[6]*frame.shape[1])
                        op = cv2.rectangle(frame,(ymax,xmax),(ymin,xmin),bgr, 2)
                        #cv2.imwrite('imagen.jpg',output)
                else:
                    op = frame
        ### TODO: Update the frame to include detected bounding boxes
        #frame = output
        # Write out the frame
        out.write(op)
        # Break if escape key pressed
        if key_pressed == 27:
            break
    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()
def main():
    args = get_args()
    infer_on_video(args)
if __name__ == "__main__":
    main()
