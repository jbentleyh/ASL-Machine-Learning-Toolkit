# American Sign Language Recognizer
Train and test a machine learning model to recognize ASL.
Datasets are not provided, but are very easy to add to the project.

# Table of Contents
* How to install
* How to integrate your own dataset
* How to train your model
* How to test your model

# How to Install
First we need to download all dependencies. It's good practice to do so inside a virtual environment, but it is not required. For more information on how to use a virtual environment, see https://virtualenv.pypa.io/en/latest/. Now, to install the dependencies, run the command:

`pip install -r requirements.txt`

Note: You may need to run an additional command: `pip install pickle-mixin`

# How to Integrate your own Dataset
By default, training and test image data is stored in `data/train` and `data/test` respectively.  The subdirectories of these two folders should consist of one folder for each letter in the alphabet, with each subfolder containing .jpg or .png files of said letter.

The intended file structure for the project goes as follows:
```
src
├── data
│   ├── train
│   │   └── A (Folder containing 'A' letter pictures)
│   │   └── B (Folder ...)
│   │   └── ...
│   ├── test
│   │   └── A
│   │   └── B
│   │   └── ...
├── datasets
│   ├── train
│   │   └── ...
│   ├── test
│   │   └── ...
├── logs
│   └── ...
├── models
│   └── ...
├── test.py
├── train.py
```
Note: I chose this data folder structure because most datasets I found online were using this folder stucture.

# How to Convert your data to the proper format
Once your folder structure is in place with your own dataset we are ready to convert our data for a format that is compatible with `train.py` and `test.py`.

To generate your training dataset, simply use the command:

`python gen_data.py`

To generate your testing dataset:

`python gen_data.py -d data/test -o datasets/test`

Note: Datasets are saved to `datasets/train` and `datasets/test` as `.csv` files.  For each line of the `.csv`, the first value is the category and the remaining `image_size^2` values are greyscale values on a scale of 0 to 1.  The first line of the `.csv` file is ignored.

# How to Train your model
Once your folder structure is in place with your own dataset we are ready to train the model.

To start training, simply use the command: 

`python train.py`

Note: Once training is complete, your models is saved to `models/`.

# How to Test your model
Once training is complete and you have your saved model `.h5` file, it's time to test it!

First, be sure to generate your test dataset using `gen_data.py`.

To test your trained model, simply use the command:

`python test.py`

# How to Evaluate your model
After testing your model, you may want even more details about your model to help you refine it.

For individual testing predictions, use the command:

`python test.py -p`

To compare your model to previous models, use the command:

`tensorboard --logdir="logs/"`

# Command line arguments
### `gen_data.py`
```
usage: gen_data.py [-h] [-s SIZE] [--min MIN] [--max MAX] [-d DATA]
                   [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  specify final image size
  --min MIN             specify canny min value
  --max MAX             specify canny max value
  -d DATA, --data DATA  path to image data
  -o OUTPUT, --output OUTPUT
                        path to save csv to
```
### `train.py`
```
usage: train.py [-h] [-d DATA] [-l LOG] [-m MODEL] [-s SIZE] [-e EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  path to training dataset
  -l LOG, --log LOG     path to log directory
  -m MODEL, --model MODEL
                        path to model directory
  -s SIZE, --size SIZE  specify training image size
  -e EPOCH, --epoch EPOCH
                        specify number of epochs
```
### `test.py`
```
usage: test.py [-h] [-d DATA] [-m MODEL] [-s SIZE] [-p]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  path to testing dataset
  -m MODEL, --model MODEL
                        path to model directory
  -s SIZE, --size SIZE  specify training image size
  -p, --predict         output redictions
```
