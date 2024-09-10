import numpy as np
import cv2
import matplotlib.cm as cm

def nms(boxes, scores, threshold=0.5):
    # Sort the bounding boxes by their confidence scores in descending order
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    
    # Initialize a list of selected bounding box indices
    selected_indices = []
    
    # Loop over the sorted list of bounding boxes
    while len(boxes) > 0:
        # Select the box or keypoint with the highest confidence score
        selected_index = indices[0]
        selected_indices.append(selected_index)
        
        # Compute the overlap between the selected box and all other boxes
        ious = compute_iou(boxes[0], boxes[1:])
        
        # Remove all boxes that overlap with the selected box by more than the threshold
        indices = indices[1:][ious <= threshold]
        boxes = boxes[1:][ious <= threshold]
    
    # Convert the list of selected box indices to an array and return it
    return np.array(selected_indices)

def compute_iou(box, boxes):
    """Compute the intersection-over-union (IoU) between a bounding box and a set of other bounding boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection
    return intersection / union

# slower, outputs all boxes
def non_max_suppression(prediction, conf_thres=0.05):
    confidences = prediction[4, :]
    conf_mask = confidences > conf_thres
    x = (prediction.T)[conf_mask]
    if x.shape[0] == 0:
        return [], [], []

    box, conf, mask = np.split(x, (4, 5), axis=1)
    boxxyxy = np.copy(box)
    boxxyxy[..., 0] = box[..., 0] - box[..., 2] / 2  # top left x
    boxxyxy[..., 1] = box[..., 1] - box[..., 3] / 2  # top left y
    boxxyxy[..., 2] = box[..., 0] + box[..., 2] / 2  # bottom right x
    boxxyxy[..., 3] = box[..., 1] + box[..., 3] / 2  # bottom right y
    box = boxxyxy

    x = np.concatenate((box, conf, mask), axis=1)

    x = x[x[:, 4].argsort()[::-1]]
    boxes, scores = x[:, :4], x[:, 4]

    indices = nms(boxes, scores)
    x = x[indices]

    return x[:, :4], x[:, 4], x[:, 5:]


def non_max_suppression_detect(prediction, conf_thres=0.05):
    confidences = prediction[4, :]
    conf_mask = confidences > conf_thres
    x = (prediction.T)[conf_mask]
    if x.shape[0] == 0:
        return [], [], []
    
    
    box, mask = x[:, :4], x[:, 4:]
    boxxyxy = np.copy(box)
    boxxyxy[..., 0] = box[..., 0] - box[..., 2] / 2  # top left x
    boxxyxy[..., 1] = box[..., 1] - box[..., 3] / 2  # top left y
    boxxyxy[..., 2] = box[..., 0] + box[..., 2] / 2  # bottom right x
    boxxyxy[..., 3] = box[..., 1] + box[..., 3] / 2  # bottom right y
    box = boxxyxy

    # breakpoint()

    conf = np.max(mask, axis=1)[...,None]
    x = np.concatenate((box, conf, mask), axis=1)

    x = x[x[:, 4].argsort()[::-1]]
    boxes, scores = x[:, :4], x[:, 4]

    indices = nms(boxes, scores)
    x = x[indices]

    return x[:, :4], x[:, 4], x[:, 5:]

def single_non_max_suppression(prediction):
    argmax = np.argmax(prediction[4,:])
    x = (prediction.T)[argmax]
    return x[:4], x[4], x[5:]

def post_process_multi(img, output, score_threshold=10):
    boxes, conf_scores, keypt_vectors = non_max_suppression(output, score_threshold)
    for keypts, conf in zip(keypt_vectors, conf_scores):
        plot_keypoints(img, keypts, score_threshold)
    return img

def post_process_single(img, output, score_threshold=10):
    box, conf, keypts = single_non_max_suppression(output)
    plot_keypoints(img, keypts, score_threshold)
    return img


sk = [15,13, 13,11, 16,14, 14,12, 11,12, 5,11, 6,12, 5,6, 5,7, 6,8, 7,9, 8,10, 1,2, 0,1, 0,2, 1,3, 2,4, 3,5, 4,6]
def plot_keypoints(img, keypoints, threshold=10):
    for i in range(0,len(sk)//2):
        pos1 = (int(keypoints[3*sk[2*i]]), int(keypoints[3*sk[2*i]+1]))
        pos2 = (int(keypoints[3*sk[2*i+1]]), int(keypoints[3*sk[2*i+1]+1]))
        conf1 = keypoints[3*sk[2*i]+2]
        conf2 = keypoints[3*sk[2*i+1]+2]

        color = (cm.jet(i/(len(sk)//2))[:3])
        color = [int(c * 255) for c in color[::-1]]
        if conf1>threshold and conf2>threshold: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(img, pos1, pos2, color, thickness=8)

    for i in range(0,len(keypoints)//3):
        x = int(keypoints[3*i])
        y = int(keypoints[3*i+1])
        conf = keypoints[3*i+2]
        if conf > threshold: # Only draw the circle if confidence is above some threshold
            cv2.circle(img, (x, y), 3, (0,0,0), -1)

def plot_boxes(results, img):
    boxes, conf_scores, classes = non_max_suppression_detect(results[0], 0.5)
    target_classes = [0, 68, 40, 66]
    class_names = ["person", "cell phone", "bottle", "remote"]

    for box, conf, cls_probs in zip(boxes, conf_scores, classes):
            cls = np.argmax(cls_probs)
            if cls in target_classes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(class_names[target_classes.index(cls)]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(img, str(round(conf, 2)), (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)