import cv2
import os
from ultralytics import YOLO
import numpy as np
import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def text_detection(image_path, model_path, output_folder):
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


def demo(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()


def main(image_path, model_path, output_folder):
    text_detection(image_path, model_path, output_folder)
    demo_args = argparse.Namespace(
        image_folder=output_folder,
        workers=4,
        batch_size=192,
        saved_model=model_path,
        batch_max_length=25,
        imgH=32,
        imgW=100,
        rgb=False,
        character='0123456789',
        sensitive=False,
        PAD=False,
        Transformation='TPS',
        FeatureExtraction='VGG',
        SequenceModeling='BiLSTM',
        Prediction='Attn',
        num_fiducial=20,
        input_channel=1,
        output_channel=512,
        hidden_size=256
    )
    demo(demo_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop objects detected in an image using YOLOv8OBB and perform text recognition")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", required=True, default="weights/Textdetect.pt", help="Path to the YOLOv8OBB model")
    parser.add_argument("--output_folder", default="demo_image", required=True, help="Path to the output folder")
    args = parser.parse_args()

    main(args.image, args.model, args.output_folder)

