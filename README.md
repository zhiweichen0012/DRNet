

<!--
 * @Description: 
 * @Author: zhiweichen
 * @Date: 2020-07-13 20:02:53
 * @LastEditTime: 2020-07-13 20:16:39
--> 
# Dilated Residual Bottleneck for Weakly Supervised Object Detection with Online Proposal Purification

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Installation](#installation)
4. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
  
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (VGG_CNN_F, VGG_CNN_M_1024), a GPU with about 6G of memory suffices.
2. For training lager networks (VGG16), you'll need a GPU with about 8G of memory.

### Installation

1. Clone the DRNet repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive git@github.com:zhiweichen95/DRNet.git
  ```

2. We'll call the directory that you cloned DRNet into `DRNet_ROOT`
 *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone DRNet with the `--recursive` flag, then you'll need to manually clone the `caffe-wsl` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-wsl` submodule needs to be on the `wsl` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $DRNet_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $DRNet_ROOT/caffe-wsl
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

6. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

7. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

8. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $DRNet_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
9. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
10. [Optional] If you want to use COCO, please see some notes under `data/README.md`
11. Follow the next sections to download pre-trained ImageNet models

### Download object proposals
1. Selective Search: [original matlab code](http://disi.unitn.it/~uijlings/MyHomepage/index.php#page=projects1), [python wrapper](https://github.com/sergeyk/selective_search_ijcv_with_python)
2. EdgeBoxes: [matlab code](https://github.com/pdollar/edges)
3. MCG: [matlab code](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/)


### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $DRNet_ROOT
./data/scripts/fetch_imagenet_models.sh
```

### Usage

To train and test a DRNet detector, use `./scripts/train_wsl.sh `.
Output is written underneath `$DRNet_ROOT/experiments`.

``Shell
cd $DRNet_ROOT
./scripts/train_wsl.sh --cfg [config] OUTPUT_DIR [output]
# config is the config you want to train with
# output is the log path 
```

Example:

```Shell
./scripts/train_wsl.sh --cfg configs/voc2007/DRNet_VGG16-C5_1x.yaml OUTPUT_DIR experiments/DRNet_VGG16-C5_1x_`date +'%Y-%m-%d_%H-%M-%S'`
```

This will reproduction the VGG16 result in paper.