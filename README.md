# Ventricle Segmentation Coding Challenge

Creating data pipeline and training pipeline for  left ventricular blood pool  (myocardium) semantic segmentation from DICOM MRI images left.  

## Install
Project is organized as a python package. So first you should install it, best with development mode.

```bash
python setup.py develop
```

Then install `PyTorch`, see [install guide](http://pytorch.org/).

## How to run it
Download data and extract it to `DATA_DIR` according to your choice.

Run [notebooks/tests.ipynb] to verify correct function of `parsing.py/polygon_to_mask()` and create data for unit testing of this method.

Run data pipline to preprocess data and split it to train/dev dataset.
```bash
python scripts/prepare_data.py
```

`DATA_DIR/dataset/` folder was created with `train/` and `dev/` subfolders containing preprocesses pickled examples prepared for training pipeline:

```bash
python scripts/train.py --train_dir DATA_DIR/dataset/train/ --dev_dir DATA_DIR/dataset//dev/ --exp exp01
```

which starts training of the net and it stores experiment data to [experiments/exp01/](experiments/exp01/) folder.


## Answers to questions

How did you verify that you are parsing the contours correctly?
 - First I converted contours to mask by untested method for some random images 
 - Then I plotted corresponding dicom and mask images to verify that mask is correct for those random images [notebooks/tests.ipynb](notebooks/tests.ipynb)
 - I stored those verified masks to test it next time without manual verification

What changes did you make to the code, if any, in order to integrate it into our production code base?
 - types in docstring 
 - changed return type in parse_dicom_file to np.ndarray
 - changed some names function and variable names
 - added some unit tests
 
Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?
 - To avoid converting contours to mask in every epoch I created [prepare_data.py](scripts/prepare_data.py) which loads whole dataset, split it to train
 and dev dataset and store each example separately in pickle file into train/dev folder.  

How do you/did you verify that the pipeline was working correctly?
 - I created [test_model_wrapper.py](tests/test_model_wrapper.py) that is checking if:
   - loading from dataset is correct  
   - trainset is shuffled for each epoch and devset not
   - loss function == log(nr_classes) before starting to train 

Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?
 - Add image enhancement in trainset (flip, rotation, adding noise, change contrast, etc.)
 - Check gradients are not zero
 - Check activations approximately equally distributed between -1 and 1 (good weight initialization) 
 - Prediction and evaluation (plot examples with the lowest/highest iou score)
 - Learning resume
 - Loading saved model
 - Automatic hyperparameter tuning
 - Better logging
 
## Notes
 - The aim of this project was to prepare data for training and create training pipeline. Training phase has not been tuned. 
 More images would probably be neccessary.
 
