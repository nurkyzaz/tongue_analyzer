# TMC-Tongue: A standardized tongue image dataset with pathological annotations for AI-assisted TCM diagnosis

Dataset DOI: [10.5061/dryad.1c59zw48r](10.5061/dryad.1c59zw48r)

## Description of the data and file structure

Using high-precision cameras to capture volunteers' tongues to obtain data, all data only includes volunteers' tongues and mouths, and does not involve volunteers' facial information. The dataset only contains photos and feature labels, and does not contain any personal privacy information of volunteers.

### Files and variables

The dataset consists of three folders, namely shezhenv3 coco, shezhenv3 txt, and shezhenv3 xml, which can be used by three different object detection models. The image data in the three folders is the same, but the label files are not the same. Users can choose the file that suits them according to their own needs. These three folders all contain test sets, training sets, and validation sets, each containing image data and corresponding label data.
shezhenv3-coco: A collection of data with tag format in coco format
Shezhenv3-txt: A collection of data with tag format in txt format
Shezhenv3 xml: A collection of data with tag format in XML format
test: test set
train: training set
val: verification set
images: folder for storing image data
labels: folder for storing label data
annotations: JSON-formatted data
classes.txt: Introduction to the names of various labels

#### File: shezhen_datasets1.zip

**Description:** Among them, there are 5594 images in the training set, 572 images in the validation set, and 553 images in the test set. Contains three annotation formats: coco/. txt/. xml, which can be used for experiments using relevant object detection algorithms through configuration files. The label data is divided into twenty categories, arranged and classified from 0 to 19, as follows: 0:jiankangshe; 1:baotaishe; 2:hongshe; 3:zishe; 4:pangdashe; 5:shoushe; 6:hongdianshe; 7:liewenshe; 8:chihenshe; 9:baitaishe; 10:huangtaishe; 11:heitaishe; 12:huataishe; 13:shenquao; 14:shenqutu; 15:gandanao; 16:gandantu; 17:piweiao; 18:xinfeitu; 19:xinfeiao;

## Code/software

After downloading, use a file decompressor to decompress the file, and then use an image viewer to view the data. You can use any universal file decompressor and image viewer to perform the above operations.

When using this dataset, it is recommended to use YOLO series models such as YOLOV8/V11/V12. The source code of these models can be downloaded directly from GitHub as open source code. Before use, please create a new YAML format file with 0: jiankangse; 1:baotaishe; 2:hongshe; 3:zishe; 4:pangdashe; 5:shoushe; 6:hongdianshe; 7:liewenshe; 8:chihenshe; 9:baitaishe; 10:huangtaishe; 11:heitaishe; 12:huataishe; 13:shenquao; 14:shenqutu; 15:gandanao; 16:gandantu; 17:piweiao; 18:xinfeitu; 19:xinfeiao; Write it in and set the path of the dataset according to your computer configuration.

## Access information

Other publicly accessible locations of the data:

[https://github.com/m28805746-max/Intelligent-tongue-diagnosis-detection-dataset](https://github.com/m28805746-max/Intelligent-tongue-diagnosis-detection-dataset)

Data was derived from the following sources:

* This dataset is our original data, collected by the device and labeled according to the doctor's advice