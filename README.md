# Machine Learning for American Sign Language
Train and test a machine learning model to recognize ASL.
Datasets are not provided, but are very easy to add to the project.

Table of Contents
==
<!--ts-->
  * [How to Install](#how-to-install)
  * [How to Integrate your own dataset](#how-to-integrate-your-own-dataset)
  * [How to Convert your Data to the Proper Format](#how-to-convert-your-data-to-the-proper-format)
  * [How to Train your Model](#how-to-train-your-model)
  * [How Do I Generate Custom Data](#how-do-i-generate-custom-data)
  * [How to Test your Model](#how-to-test-your-model)
  * [How to Train your Model](#how-to-train-your-model)
  * [Command Line Arguments](#command-line-arguments)
<!--te-->

How to Install
==

First, we need to download all dependencies. It's good practice to do so inside a virtual environment, but it is not required. For more information on how to use a virtual environment, see https://virtualenv.pypa.io/en/latest/. Now, to install the dependencies, run the command:

```
pip install -r requirements.txt
```
<br/>

Note: You may need to run an additional command: `pip install pickle-mixin`

How to Integrate your own Dataset
==

By default, training and test image data is stored in `data/train` and `data/test` respectively.  The subdirectories of these two folders should consist of one folder for each letter in the alphabet, with each subfolder containing .jpg or .png files of said letter.

The intended file structure for the project goes as follows:
```
src
├── data
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

<br/>

Note: I chose this data folder structure because most datasets I found online were using this folder stucture.

How to Convert your Data to the Proper Format
==

Once your folder structure is in place with your own dataset we are ready to convert our data for a format that is compatible with `train.py` and `test.py`.

To generate your training dataset, simply use the command:

```
python convert_data.py
```

To generate your testing dataset:

```
python convert_data.py -d data/test -o datasets/test
```

<br/>

Note: Datasets are saved to `datasets/train` and `datasets/test` as `.csv` files.  For each line of the `.csv`, the first value is the category and the remaining `image_size^2` values are greyscale values on a scale of 0 to 1.  The first line of the `.csv` file is ignored.

How Do I Generate Custom Data
==

To start generating custom training data, use the command:

```
python gen_data.py
```

To generate custom testing data, use the command:

```
python gen_data.py -o data/test
```

Position your hand so that  it is covering all nine green squares and press the '1' key.  This will sample your skin tone for background subtraction.  Next position your hand into the sign you wish to create data for.  Once in position, press the key corresponding to the sign you are holding up.  The script will save the processed frame to the correct directory multiple times per second.  While holding the sign, move it up, down, left, and right while slightly rotating in either direction.  The goal is to capture the sign at different angles.  Once you are finished generating data for that sign, press the '2' key.   Repeat this process for every character.  Press the '0' key to quit.

<br/>

Note: To generate data for the NOTHING category, press the '.' key

How to Train your Model
==

Once your folder structure is in place with your own dataset we are ready to train the model.

To start training, simply use the command: 

```
python train.py
```

<br/>

Note: Once training is complete, your models is saved to `models/`.

How to Test your Model
==

Once training is complete and you have your saved model `.h5` file, it's time to test it!

First, be sure to generate your test dataset using `gen_data.py`.

To test your trained model, simply use the command:

```
python test.py
```

How to Evaluate your model
==

After testing your model, you may want even more details about your model to help you refine it.

For individual testing predictions, use the command:

```
python test.py -p
```

To compare your model to previous models, use the command:

```
tensorboard --logdir="logs/"
```

Command Line Arguments
==

### `gen_data.py`
```
usage: gen_data.py [-h] [-s SIZE] [--min MIN] [--max MAX] [-d DATA]
                   [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  specify final image size. default=80
  --speed SPEED         specify capture speed (smaller is fater)
  --nohist              skip background subtraction
  --min MIN             specify canny min value. default=100
  --max MAX             specify canny max value. default=200
  -o OUTPUT, --output OUTPUT
                        path to save image to. default=data/train/
```

### `convert_data.py`
```
usage: convert_data.py [-h] [-s SIZE] [--min MIN] [--max MAX] [-d DATA]
                   [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  specify final image size. default=80
  --min MIN             specify canny min value. default=100
  --max MAX             specify canny max value. default=200
  -d DATA, --data DATA  path to image data. default=data/train/
  -o OUTPUT, --output OUTPUT
                        path to save csv to. default=datasets/train/
```

### `train.py`
```
usage: train.py [-h] [-d DATA] [-l LOG] [-m MODEL] [-s SIZE] [-e EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  path to training dataset. default=last modified in datasets/train/
  -l LOG, --log LOG     path to log directory. default=logs/
  -m MODEL, --model MODEL
                        path to model directory. default=models/
  -s SIZE, --size SIZE  specify training image size. default=80
  -e EPOCH, --epoch EPOCH
                        specify number of epochs. default=10
```
### `test.py`
```
usage: test.py [-h] [-d DATA] [-m MODEL] [-s SIZE] [-p]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  path to testing dataset. default=last modified in datasets/test/
  -m MODEL, --model MODEL
                        path to model directory. default=last modified in models/
  -s SIZE, --size SIZE  specify training image size. default=80
  -p, --predict         output redictions
```
