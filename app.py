import math
import gradio as gr
import easyocr
import cv2
from ultralytics import YOLO 

# Load OCR model into memory
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

# Define constants
BOX_COLORS = {
    "unchecked": (242, 48, 48),
    "checked": (38, 115, 101),
    "block": (242, 159, 5)
}
BOX_PADDING = 2

# Load models
DETECTION_MODEL = YOLO("models/detector-model.pt") 

def detect_checkbox(image_path):
    """
    Output inference image with bounding box
    Args:
    - image: to check for checkboxes
    Return: image with bounding boxes drawn and box coordinates
    """
    image = cv2.imread(image_path)
    if image is None:
        return image
    
    # Predict on image
    results = DETECTION_MODEL.predict(source=image, conf=0.1, iou=0.8) # Predict on image
    boxes = results[0].boxes # Get bounding boxes

    if len(boxes) == 0:
        return image
    
    box_coordinates = []

    # Get bounding boxes
    for box in boxes:
        detection_class_conf = round(box.conf.item(), 2)
        detection_class = list(BOX_COLORS)[int(box.cls)]
        # Get start and end points of the current box
        start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
        box = image[start_box[1]:end_box[1], start_box[0]: end_box[0], :]
        
        if detection_class == 'checked':
            box_coordinates.append((start_box, end_box))
            
            # 01. DRAW BOUNDING BOX OF OBJECT
            line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
            image = cv2.rectangle(img=image, 
                                pt1=start_box, 
                                pt2=end_box,
                                color=BOX_COLORS['checked'], 
                                thickness = line_thickness) # Draw the box with predefined colors
            
            image = cv2.putText(img=image, org=start_box, text=detection_class, fontFace=0, color=(0,0,0), fontScale=line_thickness/3)

            # 02. DRAW LABEL
            text = str(detection_class_conf)
            # Get text dimensions to draw wrapping box
            font_thickness =  max(line_thickness - 1, 1)
            (text_w, text_h), _ = cv2.getTextSize(text=text, fontFace=2, fontScale=line_thickness/3, thickness=font_thickness)
            # Draw wrapping box for text
            image = cv2.rectangle(img=image,
                                  pt1=(start_box[0], start_box[1] - text_h - BOX_PADDING*2),
                                  pt2=(start_box[0] + text_w + BOX_PADDING * 2, start_box[1]),
                                  color=BOX_COLORS['checked'],
                                  thickness=-1)
            # Put class name on image
            start_text = (start_box[0] + BOX_PADDING, start_box[1] - BOX_PADDING)
            image = cv2.putText(img=image, text=text, org=start_text, fontFace=0, color=(255,255,255), fontScale=line_thickness/3, thickness=font_thickness)
        
    return image, box_coordinates

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def nearest_coordinate(target_coord, coordinates):
    min_distance = float('inf')
    nearest_coord = None
    
    for coord in coordinates:
        distance = euclidean_distance(target_coord, coord)
        if distance < min_distance:
            min_distance = distance
            nearest_coord = coord
    
    
    return nearest_coord, euclidean_distance(target_coord, nearest_coord)

def checkbox_text_extract(image_filename):
    checkbox_img, checkbox_coordinates = detect_checkbox(image_filename)
    
    result = reader.readtext(image_filename, decoder = 'beamsearch',
                    text_threshold = 0.8, low_text = 0.2, link_threshold = 0.4,
                    canvas_size = 1500, mag_ratio = 1.5,
                    slope_ths = 0.1, ycenter_ths = 0.8, height_ths = 0.8,
                    width_ths = 1.0, y_ths = 0.8, x_ths = 1.0, add_margin = 0.1)
    
    # Get the bottom right coordinates of the CHECKED checkbox
    checkbox_bottom_right_coord = []

    for each in checkbox_coordinates:
        checkbox_bottom_right_coord.append((each[1][0], each[0][1]))

    # Sort based on the coordinates
    checkbox_bottom_right_coord = sorted(checkbox_bottom_right_coord, key=lambda point: point[1])

    detected_text = {}

    for index, each in enumerate(result):
        x_coord = int(each[0][0][0])
        y_coord = int(each[0][0][1])
        detected_text[(x_coord, y_coord)] = each[1]
        
    checked_text = ''
    for each_checkbox_coord in checkbox_bottom_right_coord:
        nearest, distance = nearest_coordinate(each_checkbox_coord, list(detected_text.keys()))
        if distance <= 15:
            checked_text += f"- {detected_text[nearest]}\n"

    return checked_text


iface = gr.Interface(fn=checkbox_text_extract, 
                     inputs=gr.Image(label="Upload image having checkboxes and text", type="filepath"), 
                     outputs=gr.Markdown())

iface.launch()