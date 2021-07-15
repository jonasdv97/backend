import cv2

def iou(boxA, boxB, image):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)
    if interArea < 0:
        interArea = 0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except:
        iou = 0
        # tl_A = (boxA[0], boxA[1])
        # br_A = (boxA[2], boxA[3])
        # color = (0, 0, 255)
        # image = cv2.rectangle(image, tl_A, br_A,color, 2)
        #
        # tl_B = (boxB[0], boxB[1])
        # br_B = (boxB[2], boxB[3])
        # color = (255, 0, 0)
        # image = cv2.rectangle(image, tl_B, br_B,color, 2)

        # print("xA: ",xA)
        # print("yA: ",yA)
        # print("xB: ",xB)
        # print("yB: ",yB)
        #
        # print('boxa: ' + str(boxAArea))
        # print('boxb: ' + str(boxBArea))
        # print('interA: '+ str(interArea))

        #cv2.imshow('Error Image', image)
        #print('Press "ESC" to end . . .')
        #if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()

    return iou