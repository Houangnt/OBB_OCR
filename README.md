## Getting Started

**Setup enviroment**
conda create -n Obb_ocr python=3.8
conda activate Obb_ocr
pip install -r requirements.txt
sudo apt-get install imagemagick
pip install ultralytics
### Run demo with pretrained model
download weights for text recogniton in here: https://drive.google.com/file/d/1_pFmvyofaXoJz74CmTbMhWZ9CHStT7vk/view?usp=sharing
python3 demo.py --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --image_folder text_crop_image/ --saved_model Textrecognition.pth --imgH 32 --imgW 128 --output_channel 128 --hidden_size 128
