[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11143184&assignment_repo_type=AssignmentRepo)
# Using pretrained CNNs for image classification
This is the third assignment of four in the Visual Analytics course

# Contribution
This assignment was made partially by myself and with help from fellow students. Especially the part on loading Json files as well as compiling the rather long main function in the end required some consultation between me and my fellow students. The helper function for plotting was also not made by myself, but provided by Ross in one of the notebooks to help plotting history plots.


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
*

## Tips

- You should not upload the data to your repo - it's around 3GB in size.
  - Instead, you should document in the README file where your data comes from, how a user should find it, and where it should be saved in order for your code to work correctly.
- The data comes already split into training, test, and validation datasets. You can use these in a ```TensorFlow``` data generator pipeline like we saw in class this week - you can see an example of that [here](https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator).
- There are a lot of images, around 106k in total. Make sure to reserve enough time for running your code!
- The image labels are in the metadata folder, stored as JSON files. These can be read into ```pandas``` using ```read_json()```. You can find the documentation for that online.
