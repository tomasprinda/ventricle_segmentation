# Ventricle Segmentation Coding Challenge

## Install
Project is organized as a python package. So first you should install it, best with development mode.

```bash
python setup.py develop
```


## Answers to questions

How did you verify that you are parsing the contours correctly?
 - First I converted contours to mask by untested method for some random images 
 - Then I plotted corresponding dicom and mask images to verify that mask is correct for those random images 
 - I stored those verified masks to test it next time without manual verification

What changes did you make to the code, if any, in order to integrate it into our production code base?
 - types in docstring 
 - changed return type in parse_dicom_file to np.ndarray
 - changed some names function and variable names
 - added some unit tests
 

Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?

How do you/did you verify that the pipeline was working correctly?

Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?