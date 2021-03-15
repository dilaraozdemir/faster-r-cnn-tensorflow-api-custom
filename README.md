# Faster R-CNN with Tensorflow API

## Creating Anaconda Environment and Requirements
	conda create -n myenv python=3.6
	conda install -c anaconda protobuf

After cloning this repo, upload from within the requirements.txt file.

    pip install -r requirements.txt
    
## Directory
### Step 1
Download the Tensorflow model file from the link below. We will do the work in this directory. Upload this repo as .zip and unzipped into the directory where you will be working.

[https://github.com/tensorflow/models](https://github.com/tensorflow/models)

**NOTE** 📝 Change the name of the file you unzipped to models.

### Step 2
Move the model in the repo, the file faster_rcnn_inception_v2_coco_2018_01_28 to the models/research/object_detection directory.

### Step 3

Run the following commands in the model/research directory.

**NOTE** 📝 Can be duplicates in command below.

    protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto  .\object_detection\protos\box_coder.proto  .\object_detection\protos\box_predictor.proto  .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto  .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\flexible_grid_anchor_generator.proto  .\object_detection\protos\calibration.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto ./object_detection/protos/anchor_generator.proto ./object_detection/protos/argmax_matcher.proto ./object_detection/protos/bipartite_matcher.proto ./object_detection/protos/box_coder.proto ./object_detection/protos/box_predictor.proto ./object_detection/protos/eval.proto ./object_detection/protos/faster_rcnn.proto ./object_detection/protos/faster_rcnn_box_coder.proto ./object_detection/protos/hyperparams.proto ./object_detection/protos/image_resizer.proto ./object_detection/protos/input_reader.proto ./object_detection/protos/losses.proto ./object_detection/protos/matcher.proto ./object_detection/protos/mean_stddev_box_coder.proto ./object_detection/protos/model.proto ./object_detection/protos/optimizer.proto ./object_detection/protos/pipeline.proto ./object_detection/protos/post_processing.proto ./object_detection/protos/preprocessor.proto ./object_detection/protos/region_similarity_calculator.proto ./object_detection/protos/square_box_coder.proto ./object_detection/protos/ssd.proto ./object_detection/protos/ssd_anchor_generator.proto ./object_detection/protos/string_int_label_map.proto ./object_detection/protos/train.proto ./object_detection/protos/keypoint_box_coder.proto ./object_detection/protos/multiscale_anchor_generator.proto ./object_detection/protos/graph_rewriter.proto ./object_detection/protos/calibration.proto ./object_detection/protos/flexible_grid_anchor_generator.proto ./object_detection/protos/center_net.proto .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto  .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\fpn.proto` ./object_detection/protos/anchor_generator.proto ./object_detection/protos/argmax_matcher.proto ./object_detection/protos/bipartite_matcher.proto ./object_detection/protos/box_coder.proto ./object_detection/protos/box_predictor.proto ./object_detection/protos/eval.proto ./object_detection/protos/faster_rcnn.proto ./object_detection/protos/faster_rcnn_box_coder.proto ./object_detection/protos/grid_anchor_generator.proto ./object_detection/protos/hyperparams.proto ./object_detection/protos/image_resizer.proto ./object_detection/protos/input_reader.proto ./object_detection/protos/losses.proto ./object_detection/protos/matcher.proto ./object_detection/protos/mean_stddev_box_coder.proto ./object_detection/protos/model.proto ./object_detection/protos/optimizer.proto ./object_detection/protos/pipeline.proto ./object_detection/protos/post_processing.proto ./object_detection/protos/preprocessor.proto ./object_detection/protos/region_similarity_calculator.proto ./object_detection/protos/square_box_coder.proto ./object_detection/protos/ssd.proto ./object_detection/protos/ssd_anchor_generator.proto ./object_detection/protos/string_int_label_map.proto ./object_detection/protos/train.proto ./object_detection/protos/keypoint_box_coder.proto ./object_detection/protos/multiscale_anchor_generator.proto ./object_detection/protos/graph_rewriter.proto ./object_detection/protos/calibration.proto ./object_detection/protos/flexible_grid_anchor_generator.proto ./object_detection/protos/center_net.proto

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
Run the code below in models/research/object_detection directory. 

    python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record

    python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record

### Step 4 : Creating labelmap.pbtxt

 - The first thing to do in this step is to move files faster_rcnn_inception_v2_pets.config, graph.pbtxt into models / research / object_detection / training directory.
 - Then a file named labelmap.pbtxt should be created in models / research / object_detection / training directory.

⚠️ **The extension of the labelmap file must be .pbtxt.**

 - 
