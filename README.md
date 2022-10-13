# Faster R-CNN with TensorFlow Object Detection API

## 📌Creating Anaconda Environment and Requirements
	conda create -n myenv python=3.6
	conda install tensorflow-gpu==1.15.0
	conda install -c anaconda protobuf
	
After cloning this repo, upload from within the requirements.txt file.

    pip install -r requirements.txt
    
## 📌Directory

### Step 1
Download the Tensorflow model file from the link below. We will do the work in this directory. Upload this repo as .zip and unzipped into the directory where you will be working.

[https://github.com/tensorflow/models](https://github.com/tensorflow/models)

**NOTE** 📝 Change the name of the file you unzipped to models.

### Step 2
Move the model in the repo, the file faster_rcnn_inception_v2_coco_2018_01_28 to the models/research/object_detection directory.

**NOTE** 📝 To find the missing files in the object_detection directory, move the missing files models/research/object_detection like in the object_detection.rar 

### Step 3
📣**Way 1:** Specify pythonpathi in system environment variables. And create new system variable 
>variable name: PYTHONPATH
>
>variable: C:\tensorflow\models;C:\tensorflow\models\research;C:\tensorflow\models\research\slim

📣**Way 2:** Set environment in Command Prompt with command below.
>SET PYTHONPATH=C:\tensorflowapi\models;C:\tensorflowapi\models\research;C:\tensorflowapi\models\research\slim

### Step 4  
Run the following commands in the model/research directory.

**NOTE** 📝 Can be duplicates in command below.

    protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto  .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\flexible_grid_anchor_generator.proto .\object_detection\protos\calibration.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\center_net.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\fpn.proto .\object_detection\protos\target_assigner.proto

Move the [setup.py](https://github.com/dilaraozdemir/repo/setup.py)  file under the model/research folder. Then run the following commands in models/research directory to run the [setup.py](https://github.com/dilaraozdemir/repo/setup.py) file we moved.

    python setup.py build
    python setup.py install

## Dataset Preparing
Your dataset must be in voc format and each image must have its own tag file (with an .xml extension).
### Step 1
Move the data you will use as test and train folders to  models/research/object_detection/images directory.
### Step 2
Run the code below in models/research/object_detection directory. In the images folder, test_labels.csv and train_labels.csv csv files will be created. You can check.

    python xml_to_csv.py
### Step 3
Type your classes in the generate_tfrecord.py file as below.

    # TO-DO replace this with label map
    def class_text_to_int(row_label):
        if row_label == 'yourclassname':
            return 1
        else:
            None

Run the code below in models/research/object_detection directory. 

    python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

    python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

### Step 4 : Creating labelmap.pbtxt

 - The first thing to do in this step is to move files faster_rcnn_inception_v2_pets.config, graph.pbtxt into models/research/object_detection/training directory.
 - Then a file named labelmap.pbtxt should be created in models/research/object_detection/training directory.

⚠️ **The extension of the labelmap file must be .pbtxt.**

 - Type your classes and ids in the labelmap.pbtxt file as below:
```
item
 {
  id: 1
  name: 'yourclassname'
 }
```
**NOTE** 📝  id = return values, name = name of your class in generate_tfrecord.py
**NOTE** 📝 Before the training models/research/object_detection/training folder must contain faster_rcnn_inception_v2_pets.config, graph.pbtxt, labelmap.pbtxt

### Step 5
Change lines in faster_rcnn_inception_v2_pets.config as below in models/research/object_detection/training directory.

 - Line 9:  # write your class count
```
    faster_rcnn {
	    num_classes: 32
```
 
 - Line 106: Write your fine_tune_checkpoint
```
    fine_tune_checkpoint : "C:/tensorflowapi/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
```
 - Line 123: Write your input path of train.record path.
```
    input_path: "C:/tensorflowapi/models/research/object_detection/train.record"
```
 - Line 125: Write your label map path. 
```
    label_map_path: "C:/tensorflowapi/models/research/object_detection/training/labelmap.pbtxt"
```
 - Line 130: Write the count of your test images in models/research/images/test folder.
```
num_examples: 1
```
 - Line 135: Write your input path of test.record path.
```
input_path: "C:/tensorflowapi/models/research/object_detection/test.record"
```
 - Line 137: Write your label map path.
```
label_map_path: "C:/tensorflowapi/models/research/object_detection/training/labelmap.pbtxt"
```

## Training
⚠️ **Before the training models/research/object_detection/inference_graph folder must be empty.**

For starting to train, run the command below in models/research/object_detection directory.

    python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

It will show like the following image.

![image](/images/training.jpg)

## Testing
### Inference Graph
⚠️ **Write the number of the last model.ckpt model created in the inference_graph directory in the XXXX part of the command.**

    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

**Example:**
The number of my model, which was formed after the training I did, was "0" as follows.

![image](/images/inference_graph.jpg)

### Test Image
### Step 1
Write the IDLE to your command and you will see screen like in the follow.

![image](/images/idlenew.jpg)

### Step 2
Choose File/Open... from the left corner of the idle. Selec the Object_detection_image.py file in directory models/research/object_detection from the screen that opens.

### Step 3
There are two options here. You can set the path of the test folder in the object_detection_image.py file, or you can move the image you want to test to the models/research / object_detection directory.

**Hint** 🗝️ Here, the operations are carried out by moving the desired image to the test models / research / object_detection directory.

    IMAGE_NAME = 'yourtestimagename.JPG'

Write the number of classes the object detector can identify.

    NUM_CLASSES = 1

### Step 4
Press F5 to Run Module.


## Maintainers

 - Dilara Ozdemir ([@GitHub dilaraozdemir](https://github.com/dilaraozdemir))
 - Buse Yaren Tekin ([@GitHub buseyarentekin](https://github.com/buseyarentekin))
 - Elif Meseci ([@GitHub elifmeseci](https://github.com/elifmeseci))
 - Süheda Cilek ([@GitHub suhedacilek](https://github.com/suhedacilek))
 - 
**NOTE** 📝  If there is a problem with the displaying Accuracy metric, you have to set up the file according to right version of Tensorflow Model.

**NOTE** 📝 You can reach from ([here](https://dilaraozdemir.medium.com/kendi-veri-k%C3%BCmeniz-ile-tensorflow-object-detection-api-kullanarak-faster-r-cnn-uygulamas%C4%B1-1e6114edf280)) the instructions in Turkish language.

