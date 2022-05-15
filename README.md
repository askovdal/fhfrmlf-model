# Title
Finding Hidden Features Responsible for Machine Learning Failures

## Contents
This repository includes all model-related scripts and files.

In order of the repository structure; 
* Bin Folder 
  * heatmap.py - for construction of heatmaps 
  * roc.py - outputs an roc graph with area under the curve score
  * roc-multi.py - outputs multiple roc graphs in one image with area under the curve scores 
  * test.py, train.py - respective scripts for testing and training the model
  * test-ensemble.py - tests the 3 best epochs of our model and returns each epoch's passed probabilities for each image
* Config Folder 
  * The .csv files are generated from scripts from our data preprocessing repository: https://github.com/askovdal/fhfrmlf 
  * config.json - includes the configuration of our model 
* data
  * dataset.py - used to load data for train.py and test.py in Bin Folder
  * imgaug.py - script for augmenting images
  * utils.py - includes utility functions such as border padding and transformation
* Model folder
  * Backbone folder
    * init.py - initialization file
    * densenet.py - includes the different densenet architectures that can be called in the config.json file
    * inception.py - includes the inception architecture --
    * vgg.py - includes the different vgg architectures --
  * attention_map.py - implements the attention map specified in the config.json file 
  * classifier.py - runs the classifier combination specified in the config.json file, giving the resulting model
  * global_pool.py - implements the different global pooling operations, our study uses PCAM
  * utils.py - includes utility functions to get normalization and optimiser types, as well as transforming tensor to numpy.
* Utils folder
  * heatmaper.py - implements the method for creating the heatmaps from the global pooling operation, our PCAM example is specified in this paper: https://arxiv.org/abs/2005.14480
  * misc.py - includes a learning rate scheduling function
* PCAM.png - An image describing the composition of the model used for our research 
* requirements.txt - A statement of what packages are needed and the minimal working version of them.

## How to reproduce our results

### Technical prerequisites
To reproduce our results, the following is needed:
* 4 CUDA GPUs
* 16 CPU cores
* +64 GB RAM  

These requirements are due to the large data size and model complexity.

### Software requirements
The following software is needed to run the scripts:
* Python 3
* pip packages listed in `requirements.txt` 

### Training the model
Run the following script using python 3. The resulting model will be saved to `logdir-50-50`
```
python3 bin/train.py config/config.json logdir-50-50 --num_workers 16 --device_ids "0,1,2,3" --verbose True
```

### Testing the model
Run the following scripts. The model will be tested on the three different experimental setups described in Section 5.1.1
```
python3 logdir-50-50/classification/bin/test.py --model_path "logdir-50-50/" --in_csv_path "config/pneumothorax-mixed-p-tube-split.csv" --out_csv_path "logdir-50-50/tube-split/test.csv" --device_ids "0,1,2,3" --num_workers 16
python3 logdir-50-50/classification/bin/test.py --model_path "logdir-50-50/" --in_csv_path "config/pneumothorax-mixed-p-w-tube.csv" --out_csv_path "logdir-50-50/w-tube/test.csv" --device_ids "0,1,2,3" --num_workers 16
python3 logdir-50-50/classification/bin/test.py --model_path "logdir-50-50/" --in_csv_path "config/pneumothorax-mixed-p-wo-tube.csv" --out_csv_path "logdir-50-50/wo-tube/test.csv" --device_ids "0,1,2,3" --num_workers 16
```

To generate the ROC curves, run the following script.
```
python3 logdir-50-50/classification/bin/roc-multi.py plot --plot_path0 "logdir-50-50/w-tube/" --plot_path1 "logdir-50-50/tube-split/" --plot_path2 "logdir-50-50/wo-tube/"
```

To generate heatmaps, create a txt file called `images.txt` containing the images you want to create heatmaps on, and run the following script. The heatmaps will be saved to the `heatmaps` folder.
```
python3 logdir-50-50/classification/bin/heatmap.py logdir-50-50/best.ckpt logdir-50-50/cfg.json images.txt heatmaps/ --device_ids '0'
```

### Testing the ensemble
When training the model, the 3 best epochs were saved as `best1.ckpt`, `best2.ckpt` and `best3.ckpt`. Run the following script to test all and save their predictions.
```
python3 logdir-50-50/classification/bin/test-ensemble.py --model_path "logdir-50-50/" --in_csv_path "config/pneumothorax-mixed-p-w-tube.csv" --out_csv_path "logdir-50-50/ensemble-w-tube/test.csv" --device_ids "0,1,2,4" --num_workers 16
```
The resulting CSV-file can be used to create a correlation matrix with the script `correlation.py` in https://github.com/askovdal/fhfrmlf.
