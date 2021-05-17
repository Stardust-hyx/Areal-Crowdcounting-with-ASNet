# Areal Crowdcounting with ASNet

Pytorch Implementation of an Areal Crowdcounting Framework based on ASNet

## Prerequisites
#### Python >= 3.6
#### Pytorch >= 1.3.1   

## Trained Model

This framework utilizes trained model's parameters provided in https://github.com/laridzhang/ASNet.
#####[Link](https://pan.baidu.com/s/1jQgBsDy90UfzlLafXgTcXQ), Password: 585s
After downloading the model, put it under the directory "./final_model".

## Data
Put images collected by camera_n under diectory "./mydata/camera_n". 
For example, in "./mydata/camera1" there is an image of resolution 1920×1080.

## Areal Partition
Put the annotation file of cutoff points under directory "./cutoff_points". The annotation file should take the following form:

```
[image's width]           [image's height]
[num of horizontal split] [num of vertical split]
[x-coord of upper end of vertical cutoff line_1] [y-coord of upper end of vertical cutoff line_1]
[x-coord of lower end of vertical cutoff line_1] [y-coord of lower end of vertical cutoff line_1]
...
[x-coord of upper end of vertical cutoff line_n] [y-coord of upper end of vertical cutoff line_n]
[x-coord of lower end of vertical cutoff line_n] [y-coord of lower end of vertical cutoff line_n]
[x-coord of left end of horizontal cutoff line_1] [y-coord of left end of horizontal cutoff line_1]
[x-coord of right end of horizontal cutoff line_1] [y-coord of right end of horizontal cutoff line_1]
...
[x-coord of left end of horizontal cutoff line_m] [y-coord of left end of horizontal cutoff line_m]
[x-coord of right end of horizontal cutoff line_m] [y-coord of right end of horizontal cutoff line_m]
```
An example 'camera1.txt' for camera1 is in "./cutoff_points". And 'camera1_annotated.jpg' is the corresponding annoatated standard image.

## Ground-truth
Put the file of ground-truth counts for camera_n under diectory "./mydata/camera_n". It should take the following form:
```
[area_1 count for img_1] [area_2 count for img_1] ...
...
[area_1 count for img_N] [area_2 count for img_N] ...
```
, where areas are indexed from left to right and from top to buttom.  
An example 'camera1_count.txt' for camera1 is in "./mydata/camera1".

##  Testing
To test our framwork on images collected by camera1, run
```
python test.py
```
Result will be saved as 'camera1_result.txt'.

