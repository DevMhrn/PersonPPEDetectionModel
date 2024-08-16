import os
import xml.etree.ElementTree as ET
import shutil
import argparse

# Class names array
classes = ['person', 'hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']

# Function to convert PascalVOC bbox to YOLO format
def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

# Function to process each XML file
def process_xml(xml_file, base_input_dir, person_detection_dir, ppe_detection_dir, classes, ppe_class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    has_person = False
    has_ppe = False
    person_labels = []
    ppe_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name in classes:
            bbox = obj.find('bndbox')
            b = (float(bbox.find('xmin').text), float(bbox.find('xmax').text),
                 float(bbox.find('ymin').text), float(bbox.find('ymax').text))
            yolo_bbox = convert_bbox_to_yolo((width, height), b)

            if class_name == 'person':
                person_labels.append(f"{classes.index(class_name)} {' '.join(map(str, yolo_bbox))}\n")
                has_person = True
            else:
                # Map to new index in ppe_class_mapping
                new_index = ppe_class_mapping[class_name]
                ppe_labels.append(f"{new_index} {' '.join(map(str, yolo_bbox))}\n")
                has_ppe = True

    base_filename = os.path.splitext(os.path.basename(xml_file))[0]
    
    # If the image contains a person, save it to the person detection folder
    if has_person:
        shutil.copy(os.path.join(base_input_dir, "images", f"{base_filename}.jpg"), os.path.join(person_detection_dir, "images", f"{base_filename}.jpg"))
        with open(os.path.join(person_detection_dir, "labels", f"{base_filename}.txt"), 'w') as f:
            f.writelines(person_labels)
    
    # If the image contains PPE classes, save it to the PPE detection folder
    if has_ppe:
        shutil.copy(os.path.join(base_input_dir, "images", f"{base_filename}.jpg"), os.path.join(ppe_detection_dir, "images", f"{base_filename}.jpg"))
        with open(os.path.join(ppe_detection_dir, "labels", f"{base_filename}.txt"), 'w') as f:
            f.writelines(ppe_labels)

def main():
    parser = argparse.ArgumentParser(description="Filter PascalVOC annotations and convert to YOLO format for person and PPE detection.")
    parser.add_argument('--input_dir', type=str, required=True, help='Base input directory containing "images" and "labels" subdirectories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory where YOLO annotations will be saved.')

    args = parser.parse_args()

    person_detection_dir = os.path.join(args.output_dir, "person_detection")
    ppe_detection_dir = os.path.join(args.output_dir, "ppe_detection")
    
    # Ensure the output directories exist
    os.makedirs(os.path.join(person_detection_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(person_detection_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(ppe_detection_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(ppe_detection_dir, "labels"), exist_ok=True)
    
    # Create PPE class mapping (excluding 'person') and reindex from 0
    ppe_classes = [cls for cls in classes if cls != 'person']
    ppe_class_mapping = {cls_name: i for i, cls_name in enumerate(ppe_classes)}
    
    # Iterate through all XML files and process them
    for file in os.listdir(os.path.join(args.input_dir, "labels")):
        if file.endswith(".xml"):
            process_xml(os.path.join(args.input_dir, "labels", file), args.input_dir, person_detection_dir, ppe_detection_dir, classes, ppe_class_mapping)

if __name__ == "__main__":
    main()
