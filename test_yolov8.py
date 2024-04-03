from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def rotate_if_vertical(image_folder):
    for filename in os.listdir(image_folder):
        # Đọc ảnh từ thư mục
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
        # Kiểm tra nếu ảnh là hình chữ nhật đứng
        height, width = image.shape[:2]
        print("H W : %d %d" % (height, width))
        if height > width:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # Ghi đè lên tệp ảnh gốc
            cv2.imwrite(image_path, rotated_image)
            print(f"Rotated {filename} and saved as {image_path}")
        else:
            print(f"{filename} is not vertical. No rotation needed.")
def rotate_and_crop_images(image_folder):
    output_folder = "cropped_images"
    os.makedirs(output_folder, exist_ok=True)

    model = YOLO(model='/content/model/ocr.pt', task="obb")

    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)

            results = model.predict(image)

            def sort_points(points):
                s = points.sum(axis=1)
                diff = np.diff(points, axis=1)
                sorted_points = np.zeros((4, 2), dtype="float32")

                sorted_points[0] = points[np.argmin(s)]
                sorted_points[2] = points[np.argmax(s)]
                sorted_points[1] = points[np.argmin(diff)]
                sorted_points[3] = points[np.argmax(diff)]

                return sorted_points

            def crop_rotated_rectangle_with_points(image, points):
                points = np.array(points, dtype="float32")
                sorted_points = sort_points(points)
                (tl, bl, br, tr) = sorted_points
                width = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
                height = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))

                dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')

                M = cv2.getPerspectiveTransform(sorted_points, dst)

                warped = cv2.warpPerspective(image, M, (width, height))

                return warped

            for result in results:
                bounding_boxes = result.obb.xyxyxyxy.numpy()
                for i, bounding_box in enumerate(bounding_boxes):
                    points = bounding_box.reshape(4, 2)
                    cropped_image = crop_rotated_rectangle_with_points(image, points)
                    cropped_image_name = f"{os.path.splitext(image_name)[0]}_cropped_{i}.jpg"
                    output_file = os.path.join(output_folder, cropped_image_name)
                    cv2.imwrite(output_file, cropped_image)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(cropped_image)
                    plt.axis('off')
                    plt.show()

    rotate_if_vertical(output_folder)

rotate_and_crop_images("/content/test_img")
