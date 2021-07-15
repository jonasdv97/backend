#print("Always executed")

import cv2
import xml.etree.ElementTree as ET


def read_image(image_location):
    return cv2.imread(image_location)

def write_image(image_location, image):
    cv2.imwrite(image_location, image)

def parse_annotation(ann_path, labels=[]):
    all_imgs = []
    seen_labels = {}
    img = {'object': []}
    tree = ET.parse(ann_path)
    cellnbr=0
    for elem in tree.iter():

        if 'width' in elem.tag:
            img['width'] = int(elem.text)
        if 'height' in elem.tag:
            img['height'] = int(elem.text)
        if 'filename' in elem.tag:
            img['filename'] = elem.text
        if 'object' in elem.tag or 'part' in elem.tag:
            obj = {}

            for attr in list(elem):

                if 'name' in attr.tag:
                    cellnbr+=1
                    obj['name'] = attr.text

                    if obj['name'] in seen_labels:
                        seen_labels[obj['name']] += 1
                    else:
                        seen_labels[obj['name']] = 1

                    if len(labels) > 0 and obj['name'] not in labels:
                        break
                    else:
                        obj['name'] = attr.text+str (cellnbr)
                        img['object'] += [obj]

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))

    if len(img['object']) > 0:
        all_imgs += [img]

    return all_imgs, seen_labels

def AddAnnotationToImage(image,image_annotation):
    annotation_data = image_annotation[0][0]

    for cell in annotation_data["object"]:
        #print(cell)
        cv2.rectangle(image,(cell['xmin'],cell['ymin']),(cell['xmax'],cell['ymax']),(0,255,0),2)
        cv2.putText(image,cell['name'],(cell['xmin']+10,cell['ymin']+10),0,0.3,(0,0,255))

    return image

def readImageWithAnnotation(image_path, annotation_path):
    image = read_image(image_path)
    annotation = parse_annotation(annotation_path)

    image_processed = AddAnnotationToImage(image, annotation)

    return image_processed

def create_grayScaleImage(image_path):
    image = read_image(image_path)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayImage

def create_blackAndWhiteImage(image_path):
    grayImage = create_grayScaleImage(image_path)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    return blackAndWhiteImage


if __name__ == '__main__':
    import os
    abs_path_img7=os.path.join(os.path.abspath("../"),"Data","Original_Images","BloodImage_00007.jpg")
    rel_path_img7=os.path.relpath(abs_path_img7,"")
    print("Absolute path: ",abs_path_img7)
    print("Relative path: ",rel_path_img7)
    img7 = read_image(abs_path_img7)


    #print(img7)

