# How to make it work:

Put .py scripts in directory with folders named 'train_images' and 'base'.
'train_images' will consist of images you want a model to train on and 'base' will have images the model will be tested on.

You can use model included in this repository or make your own.

Run **model_creation.py** to train the model and create .csv table consistng of keypoint descriptors of images located in 'base'.

To see the actual result use ***streamlit run sim_pics.py***

# Example:
![example](https://user-images.githubusercontent.com/81494616/208318692-fae44f26-9ea6-4b1e-bdf3-3c536bf90553.png)
