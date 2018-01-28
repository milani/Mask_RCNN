import os
import time
import numpy as np

from config import Config
import utils
import model as modellib
from glob import glob
import skimage.io
import skimage.transform
import scipy.misc
import pandas as pd

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
NUCLEI_MODEL_PATH = os.path.join(ROOT_DIR, "nuclei.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class NucleiConfig(Config):
    NAME = "nuclei"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1  # Background + Nuclei
    USE_MINI_MASK=True
    MINI_MASK_SHAPE = (320,320)
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    RPN_NMS_THRESHOLD = 0.7
    IMAGE_MIN_DIM =400
    IMAGE_MAX_DIM=640


############################################################
#  Dataset
############################################################

class NucleiDataset(utils.Dataset):
    def load_nuclei(self, dataset_dir, subset, stage):
        self.add_class("nuclei",1,'NUCLEI')

        extract_id_pattern = lambda path: path.split('/')[-1]
        image_dir = "{}/{}_{}".format(dataset_dir, stage, subset)
        paths = glob(os.path.join(image_dir,'*'))
        image_ids = list(map(extract_id_pattern,paths))

        # Add images
        for i in image_ids:
            self.add_image(
                "nuclei", image_id=i,
                path=os.path.join(image_dir, i, 'images', i+'.png' )
                #width=,
                #height=
            )
        print("image list size:",len(self.image_info))
    
    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.shape[2] == 4:
            image = image[:,:,:3]
        return image

    def load_mask(self, image_id):
        image_path = self.image_info[image_id]['path']
        extract_path = lambda path: path.split('/')[:-2]
        paths = extract_path(image_path)
        paths = glob(os.path.join(*paths,'masks','*'))
        masks = skimage.io.imread_collection(paths).concatenate()
        #masks = np.swapaxes(masks,0,2)
        masks = np.moveaxis(masks,0,-1)
        masks = np.where(masks > 0, True, False)
        class_ids = np.array([1]*masks.shape[2],dtype=np.int32)
        return masks,class_ids

    def image_reference(self, image_id):
        return "[empty]"


############################################################
#  Evaluation
############################################################
def rle_encode(x):
    dots = np.where(x.T.flatten()==True)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def remove_overlap(masks):
    masks = masks.copy()
    overlap_pixels = np.where(masks.sum(axis=2)>1)
    maxs = np.argmax(masks[overlap_pixels[0],overlap_pixels[1],:],axis=1)
    for i,j,z in zip(overlap_pixels[0],overlap_pixels[1],maxs):
        masks[i,j,:] = 0
        masks[i,j,z] = 1
    empty_masks = []
    for i in range(masks.shape[2]):
        if masks[:,:,i].sum() == 0:
            empty_masks.append(i)

    return np.delete(masks,empty_masks,axis=2)


def build_results(dataset, image_ids, rois, class_ids, scores, masks):
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    masks = remove_overlap(masks)

    for image_id in image_ids:
        # Loop through detections
        for i in range(masks.shape[2]):
            mask = masks[:, :, i]
            result = {
                "ImageId": image_id,
                "EncodedPixels": str(rle_encode(mask))[1:-1].replace(',','')
            }
            results.append(result)
    return results


def generate_submission(model, dataset):
    image_ids = dataset.image_ids

    nuclei_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        image_results = build_results(dataset, nuclei_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)
    out = pd.DataFrame(results)
    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    return out


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Nuclei dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/data/",
                        help='Directory of the dataset')
    parser.add_argument('--stage', required=False,
                        default='stage1',
                        metavar="<stage>",
                        help='Stage')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--out', required=False,
                        default='submission.csv',
                        metavar="/path/to.csv",
                        help='Output path for submission csv file')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Stage: ", args.stage)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleiConfig()
    else:
        class InferenceConfig(NucleiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    #else:
    #    model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = NucleiDataset()
        dataset_train.load_nuclei(args.dataset, "train", stage=args.stage)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = NucleiDataset()
        dataset_val.load_nuclei(args.dataset, "validation", stage=args.stage)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        #print("Training network heads")
        #model.train(dataset_train, dataset_val,
        #            learning_rate=config.LEARNING_RATE,
        #            epochs=40,
        #            layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        #print("Fine tune Resnet stage 4 and up")
        #model.train(dataset_train, dataset_val,
        #            learning_rate=config.LEARNING_RATE,
        #            epochs=120,
        #            layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = NucleiDataset()
        nuclei = dataset_val.load_nuclei(args.dataset, "validation", stage=args.stage)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_nuclei(model, dataset_val, nuclei, "bbox", limit=int(args.limit))
    elif args.command == 'submit':
        dataset_test = NucleiDataset()
        nuclei = dataset_test.load_nuclei(args.dataset, "test", stage=args.stage)
        dataset_test.prepare()
        print("Running submission on {} images.".format(len(dataset_test.image_ids)))
        out = generate_submission(model, dataset_test)
        out.to_csv(args.out,index=False,columns=['ImageId','EncodedPixels'])
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
