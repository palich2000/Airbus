import os
import argparse
import time
import numpy as np
import cv2
import tensorflow as tf
import torch
import torchvision

def xywh2xyxy(x):
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]

    return output


def slice_image(image_path, slice_size=512):
    img = cv2.imread(image_path)
    slices = []
    height, width = img.shape[:2]

    for x in range(0, width, slice_size):
        line = []
        for y in range(0, height, slice_size):
            slice_img = img[y:y + slice_size, x:x + slice_size]
            line.append(np.ascontiguousarray(slice_img))
        slices.append(line)
    return slices


def merge_images(slices):
    img = None
    for line in slices:
        line_img = None
        for slice_img in line:
            if line_img is None:
                line_img = slice_img
            else:
                line_img = np.concatenate((line_img, slice_img), axis=0)
        if img is None:
            img = line_img
        else:
            img = np.concatenate((img, line_img), axis=1)
    return img


def detect(interpreter, image, treshold):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    # TODO check shape of input_shape and frame.shape
    _, w, h, _ = input_shape

    input_img = image[np.newaxis, ...]
    input_img = input_img.astype(np.float32) / 255.  # change to float img
    interpreter.set_tensor(input_details[0]['index'], input_img)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    preds = interpreter.get_tensor(output_details[0]['index'])
    preds = torch.from_numpy(preds)
    preds = non_max_suppression(preds,
                                treshold,
                                0.7,  # todo, make into arg
                                agnostic=False,
                                max_det=300,
                                classes=None)  # hack. just copied values from execution of yolov8n.pt

    results = []
    for i, pred in enumerate(preds):
        # print("pred OUT {} \npred:{} \nbox:{}".format(i, pred, pred[:, :4]))
        # orig_img = orig_imgs[i]
        #
        #
        # pred[:, :4] = ops.scale_boxes(input_img.shape[1:], pred[:, :4], orig_img.shape)
        # img_path = ""
        results.append(pred)

    print('inference time: %.2f ms' % (inference_time * 1000))
    return results


def draw_boxes(image, boxes, labels):
    if not boxes:
        return
    w, h = image.shape[1], image.shape[0]
    # print(f"image shape: {w}, {h}")
    i = 0
    for bl in boxes:
        for box in bl:
            if len(box) == 0:
                continue
            # print(f"box {i}: {box}")
            xmin, ymin, xmax, ymax = box[:4]
            xmin, ymin, xmax, ymax = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
            # print(f"box {i}: {xmin}, {ymin}, {xmax}, {ymax}")
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            i += 1


def process_image(interpreter, image_path, labels, threshold):
    slices = slice_image(image_path)
    i = 0
    for line in slices:
        j = 0
        for slice_img in line:
            boxes = detect(interpreter, slice_img, threshold)
            draw_boxes(slice_img, boxes, labels)
            cv2.putText(slice_img, f"({i},{j})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(slice_img, (0, 0), (512, 512), (128, 128, 128), 1)
            j += 1
        i += 1
    return merge_images(slices)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='File path of .tflite file')
    parser.add_argument('-i', '--input', required=True,
                        help='File path of image to process')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-t', '--threshold', type=float, default=0.2,
                        help='Score threshold for detected objects')
    args = parser.parse_args()

    labels = read_label_file(args.labels) if args.labels else {}
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    input_shape = input_details[0]['shape']
    print('input_shape: ', input_shape)
    _, w, h, _ = input_shape
    print('w: ', w, 'h: ', h)
    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
    if os.path.isfile(args.input):
        print(f"Processing {args.input}")
        cv2.imshow('frame', process_image(interpreter, args.input, labels, args.threshold))
        while cv2.waitKey(1) & 0xFF != ord('q'):
            pass
    elif os.path.isdir(args.input):
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(root, file)
                    print(f"Processing {image_path}")
                    img = process_image(interpreter, image_path, labels, args.threshold)
                    # img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('frame', img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        return 0
        while cv2.waitKey(1) & 0xFF != ord('q'):
            pass


if __name__ == '__main__':
    main()
