import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy
import time as gettime

warnings.filterwarnings("ignore")


class ONNXDetect:
    def __init__(self, args, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])

        self.inputs = self.session.get_inputs()[0]
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.7
        self.input_size = args.input_size
        shape = (1, 3, self.input_size, self.input_size)
        image = numpy.zeros(shape, dtype='float32')
        for _ in range(10):
            self.session.run(output_names=None,
                             input_feed={self.inputs.name: image})

    def __call__(self, image):
        image, scale = self.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))[::-1]
        image = image.astype('float32') / 255
        image = image[numpy.newaxis, ...]

        outputs = self.session.run(output_names=None,
                                   input_feed={self.inputs.name: image})
        outputs = numpy.transpose(numpy.squeeze(outputs[0]))

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_indices = []

        # Iterate over each row in the outputs array
        for i in range(outputs.shape[0]):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = numpy.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_threshold:
                # Get the class ID with the highest score
                class_id = numpy.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                image, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((image - w / 2) / scale)
                top = int((y - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)

                # Add the class ID, score, and box coordinates to the respective lists
                class_indices.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)

        # Iterate over the selected indices after non-maximum suppression
        nms_outputs = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_indices[i]
            nms_outputs.append([*box, score, class_id])
        return nms_outputs

    @staticmethod
    def resize(image, input_size):
        shape = image.shape

        ratio = float(shape[0]) / shape[1]
        if ratio > 1:
            h = input_size
            w = int(h / ratio)
        else:
            w = input_size
            h = int(w * ratio)
        scale = float(h) / shape[0]
        resized_image = cv2.resize(image, (w, h))
        det_image = numpy.zeros((input_size, input_size, 3), dtype=numpy.uint8)
        det_image[:h, :w, :] = resized_image
        return det_image, scale

def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=160, type=int)
    args = parser.parse_args()
    
    # Load model
    model = ONNXDetect(args, onnx_path='weights/yolov8_160_0.29.onnx')
    
    for i in range(1, 13):
        input_file = f'image/inferensi ({i}).jpg'
        output_file = f'inference time/inferensi ({i})_yolov8_160.jpg'
        
        frame = cv2.imread(input_file)
        image = frame.copy()
        
        start = gettime.perf_counter()
        outputs = model(image)
        end = gettime.perf_counter()
        inference_time = (end - start) * 1000.
        print(f"Inference time for {input_file}: {inference_time}ms")
        
        
        labels = ["Normal", "Karies kecil", "Karies sedang", "Karies besar", "Stain", "Karang gigi", "Lain-Lain"]
        
        label_colors = {
            "Normal": (0, 255, 0),
            "Karies kecil": (0, 0, 255),
            "Karies sedang": (0, 0, 130),
            "Karies besar": (0, 0, 50),
            "Stain": (255, 0, 255),
            "Karang gigi": (0, 255, 255),
            "Lain-Lain": (128, 128, 128)
        }
        
        text_size, _ = cv2.getTextSize('Inference time: %.2fms' % inference_time, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        box_height = text_size[1] + 30  # Increase the height of the box
        cv2.rectangle(frame, (0, 0), (10 + text_size[0] + 10, 10 + box_height), (0, 0, 0), -1)
        cv2.putText(frame, 'Inference time: %.2fms' % inference_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        for output in outputs:
            x, y, w, h, score, index = output
            label = labels[index]  # Replace "Unknown" with the actual label based on the index
            color = label_colors.get(label, (0, 0, 0))  # Get the color based on the label
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        cv2.imwrite(output_file, frame)


if __name__ == "__main__":
    main()
