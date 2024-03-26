import cv2
import os
from ultralytics import YOLO
import numpy as np

def crop_rotated_rectangle_with_points(image, points):
    def sort_points(points):
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        sorted_points = np.zeros((4, 2), dtype="float32")
        sorted_points[0] = points[np.argmin(s)]
        sorted_points[1] = points[np.argmax(s)]
        sorted_points[2] = points[np.argmin(diff)]
        sorted_points[3] = points[np.argmax(diff)]
        return sorted_points

    points = np.array(points, dtype="float32")
    sorted_points = sort_points(points)
    (tl, bl, br, tr) = sorted_points
    width = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    height = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(sorted_points, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def main(image_path, model_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image = cv2.imread(image_path)
    model = YOLO(model=model_path, task="obb")
    results = model.predict(image)
    for result in results:
        bounding_boxes = result.obb.xyxyxyxy.numpy()
        for i, bounding_box in enumerate(bounding_boxes):
            points = bounding_box.reshape(4, 2)
            cropped_image = crop_rotated_rectangle_with_points(image, points)
            cropped_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_cropped_{i}.jpg"
            output_file = os.path.join(output_folder, cropped_image_name)
            cv2.imwrite(output_file, cropped_image)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop objects detected in an image using YOLOv8OBB")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", required=True, default="weights/Textdetect.pt", help="Path to the YOLOv8OBB model")
    parser.add_argument("--output_folder", default="demo_image", required=True, help="Path to the output folder")
    args = parser.parse_args()
    main(args.image, args.model, args.output_folder)
