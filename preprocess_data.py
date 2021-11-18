from pathlib import Path
from shutil import copyfile
import argparse


ORIGINAL_PATH = Path('coco/train2017/')
DESTINATION_IMAGE_PATH = Path('coco/images/train2017')
DESTINATION_LABEL_PATH = Path('coco/labels/train2017')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="relocate dataset for training")
    parser.add_argument('--original_path', type=Path, default=ORIGINAL_PATH)
    parser.add_argument('--destination_image_path', type=Path, default=DESTINATION_IMAGE_PATH)
    parser.add_argument('--destination_label_path', type=Path, default=DESTINATION_LABEL_PATH)
    args = parser.parse_args()
    
    dataset_path = args.original_path
    images = list(dataset_path.glob("*.jpg"))
    labels = list(dataset_path.glob("*.txt"))
    image_path = args.destination_image_path
    image_path.mkdir(parents=True, exist_ok=True)
    label_path = args.destination_label_path
    label_path.mkdir(parents=True, exist_ok=True)
    
    for x in images:
        destination = image_path/x.name
        copyfile(x, destination)
    
    for x in labels:
        destination = label_path/x.name
        copyfile(x, destination)