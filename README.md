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
Before we add our dataset, we need to create a `Train` and a `Test` folder within our source folder. Within these new folders will contain a folder for each letter in the alphabet. The letter folders with contain .jpg or .png pictures of said letter for training and testing.

The intended file structure for the project goes as follows:
```
src
├── train.py
├── test.py
├── Train
│   └── A (Folder containing 'A' letter pictures)
│   └── B (Folder ...)
│   └── ...
├── Test
│   └── A (Folder containing 'A' letter pictures)
│   └── B (Folder ... )
│   └── ...
```
Note: I chose this folder structure because most datasets I found online were using this folder stucture.

# How to Train your model
Once your folder structure is in place with your own dataset we are ready to train the model.

To start training, simply use the command: 

`python train.py`

Note: Once training is complete, your model should saved as `sign_lang_model.h5`.

# How to Test your model
Once training is complete and you have your saved model `.h5` file, it's time to test it!

First, add your test pictures for each letter in the letter folders within the `Test` folder.

To test your trained model, simply use the command:

`python test.py`
