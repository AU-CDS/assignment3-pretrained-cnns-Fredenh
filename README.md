[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11143184&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Using pretrained CNNs for image classification
This is the third assignment of four in the Visual Analytics course

# Contribution
This assignment was made partially by myself and with a great amount of help from my fellow students. It was an assignmnent where we sparred a lot on our code to make it work in the end. Especially the part on loading Json files as well as compiling the rather long main function in the end required some consultation between me and my fellow students. The helper function for plotting was also not made by myself, but provided by Ross in one of the notebooks to help plotting history plots. The helper function is located in the _utils_ folder. The skill level required for this assignment is a bit higher than the previous ones, which makes it more necessaray to consult the internet for help in setting up ones script. For reading in the Json files from the _metadata_ folder, i consulted a [link](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html) that Ross provided and I adapted my code to that. Similarly with the datagenerator portion of the pipeline, i took inspiration from the following [tutorial](https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c) on how to set up the datagenerator for train, test and validation using the ```flow_from_dataframe()``` function.


# Ross' instructions
In the previous assignments involving classification, we were performing a kind of simple feature extraction on images by making them greyscale and flattening them to a single vector. This vector of pixel values was then used as the input for some kind of classification model.

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which trains a classifier on this dataset using a *pretrained CNN like VGG16*
- Save the training and validation history plots
- Save the classification report

# Data
The data for this assignment is the [Indo Fashion dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) by Kaggle user Rashmi Margani. It is a dataset that consists of 106K images and 15 unique cloth categories. The creators describe the dataset in the following way "For a fair evaluation, we ensure equal distribution of the classes in the validation and the test set consisting of 500 samples per class." (Margani 2023). 

# Packages 
I used a larger variety of packages for this assignment since the complexity of it is also higher than the previous assignments. I will in the following bulletpoints list them and for what purpose they were needed
* ```os``` for path management
* ```tensorflow``` is used for model tracking and training
* ```scikit-learn```is used to build classifiers on the image classification data
* ```Numpy``` is used for mapping the predictions. It also plays a part in the helper function _plot_history()_ 
* ```matplotlib``` is used for plotting 
* ```Pandas```is used for creating dataframes and making samples of the training and validation data and reading in Json files
* ```Json``` is used for working with Json files
* ```sys``` is used to navigate the directories

# Methods
The code for this assignment is located in the _src_ folder. There are two seperate scripts. One called _a3short.py_ and one called _a3code.py_. The first of the two is the version i have run from the command line and gotten an output from. In the script i run a subset of the data for the sake of quicker model finetuning. I created the subset of the data using the _samples_ method from ```Pandas```. However, i have provided an example of how the pipeline should look if one wants to run the finetuning of the pretrained model on all the data (which is 106k images, so that would take a while). That example is the _a3code.py_ script.

Now i will run through what the script does. Firstly it reads the Json files from the _metadata_ folder with the help of ```read_json()``` from ```pandas```. Then the subset is created and the _ImageDataGenerator()_ is initialized before the ```flow_from_dataframe()```, mentioned previously, is utilized in the pipeline in order to allow me to input a Pandas dataframe containing the filenames and class names and read the images directly from the directory with their respective class names mapped. Then the script starts building the classifier. For this assignment, the VGG16 model is utilized and finetuned on the Indo Fashion data. In setting up the model, i choose to freeze the weights because it retains the knowledge that the model has and lets me leverage it for the finetuning purposes. It also prevents the model from overfitting. The images are also flattened before the new model is defined and compiled. Then the Indo Fashion training data is fit to the new model in order to start the finetuning. Then the _main()_ function is compiled where the training and validation history plots are saved using the helper function imported from the _utils_ folder. The classification report is also printed in the _main()_ function. Both the outputs are located in the _out_ folder.
 
# Discussion of results
The output from the finetuning on a subset of the data resulted in a f1-score of 0.77, which is very good considering the rather small train and val size of 10000 and 5000. This shows the functionality of using pretrained models for finetuning image classification data. It is an approach that is computatonally cheaper than compiling a complex model from the bottom and training it on a particular dataset. It satisfies the ethics of programming in the present where there is an increasing emphasis on carbon footprint. Although the finetuning of all the metadata would take many hours, it would still be more effective than compiling a huge model a single purpose. It is clear that the model performed better on certain labels. For instance, the _blouse_ class is easily predicted by the model with an accuracy of 0.93. The same cannot be said for _gowns_ where the accuracy score is as low as 0.57. So there are inconsistencies in the model's performance, but the overall performance is satisfactory. 
The constant downward trajectory of the loss plot indicates that the model learned between each epoch and therefore made better predictions. However, the loss function has a steeper gradient in the beginning and then it evens out slightly between the 6th and 8th epoch. It shows that the model was quick to learn and then the accuracy became so high that only minor improvements were possible towards the latter half of the epochs. For the validation plot, the trajectory of the training and validation accuracy indicate that there is no big signs of overfitting. The validation accuracy plateaus a little bit between epoch 4 and 6, but still improves slightly. Overall, it can be concluded that the finetuning of the VGG16 model on a subset of the Indo Fashion was sucessfull and created a good insight into how useful a tool pretrained models are.

# Usage
* First you need to acquire the data from [Kaggle](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset?select=val_data.json) and place it in the _in_ folder
* Then you run ```bash setup.sh``` from the command line to install requirements and create the virtual environment
* Then Run source ```./assignment3_env/bin/activate``` to activate the virtual environment
* To run the script with the subset of the data run ```python3 src/a3short.py``` from the command line
* If you want to run the script that finetunes on all the data then run ```python3 src/a3code.py``` from the command line
* After running the script the classification report and the loss- and validation plots are located in the _out_ foler
