import os
import argparse

def allFileList(rootfile, allFile):
    folder = os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile, temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName, allFile)

def is_str_right(plate_name):
    for str_ in plate_name:
        if not str_.isdigit():
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="/mnt/EPan/carPlate/@realTest2_noTraining/realrealTest", help='source')
    parser.add_argument('--label_file', type=str, default='datasets/val.txt', help='model.pt path(s)')

    opt = parser.parse_args()
    rootPath = opt.image_path
    labelFile = opt.label_file

    fp = open(labelFile, "w", encoding="utf-8")
    file = []
    allFileList(rootPath, file)
    picNum = 0

    for jpgFile in file:
        jpgName = os.path.basename(jpgFile)
        name, ext = os.path.splitext(jpgName)
        if not name.isdigit() or ext.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        labelStr = "\t"  # Use tab instead of space
        strList = list(name)
        for i in range(len(strList)):
            labelStr += strList[i]

        picNum += 1
        fp.write(jpgFile + labelStr + "\n")

    fp.close()