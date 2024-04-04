## Getting Started

**Setup enviroment** <br>
conda create -n Obb_ocr python=3.10 <br>
conda activate Obb_ocr <br>
pip install -r requirements.txt <br>
pip install ultralytics <br>
### Run demo with my model
download weight for text detection in here: https://drive.google.com/file/d/1nGKdeEYGRi07_aIjR-PlqEi-XlDkMnlh/view?usp=drive_link <br>
download weights for text recogniton in here: https://drive.google.com/file/d/1_pFmvyofaXoJz74CmTbMhWZ9CHStT7vk/view?usp=sharing <br>
python3 demo.py --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --image_folder text_crop_image/ --saved_model Textrecognition.pth --imgH 32 --imgW 128 --output_channel 128 --hidden_size 128
