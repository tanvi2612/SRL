# Semantic Role Labeling

An attempt to create a Semantic Role Labeler for Hindi using Supervised Learning approach. This model is divided into two tasks of argument identification and argument classification, both handled using Support Vector Machines. 
We use features such as chunk tags, head word of the chunk, POS-tag of the head word, dependency labels, postpositions etc. and a combination of these features to identify the correct semantic role of each chunk.


## Dataset
The dataset used to train both the SVMs is the **Hindi PropBank** which contains approximately 1100 chunked sentences in the ssf format. We also use *Fasttext* to convert the head word into vectors.


---

## How to Run the Code
Pre-requisites:
1) Jupyter-notebook
2) Python 3.6 and above

Input to the code must be in csv format similar to what can be found in the file expected.csv

1) Open Hindi_SRL.ipynb
2) Run the kernel

---
## Contribution by the team members
Each team member has contributed equally to the project. Initially both of us put equal efforts towards building the dataset and preprocessing. Sumanth Balaji took care of the first part of Argument Identification and Tanvi Kamble created the model for the second part i.e. Argument Classification.

---
##Feature analysis
For both tasks of the approach used (i.e argument identification and argument classification) we make use of features such as chunk tags, head word of the chunk, POS-tag of the head word, dependency labels, postpositions etc. and a combination of these features. Of the combinations of features analysed it is found that the combination of features (Head word, Dependency, Dependency head) produced the best results for both the tasks. The folder outputs and outputs2 contains the feature analysis report for each combination of features in correspondingly named text files. 

We had also tried the different kernels available for SVM classifier and from analysis we reach the conclusion that the Linear kernel works best with the tasks we are dealing with

## Github repository
https://github.com/tanvi2612/SRL

The dataset, outputs, the different models trained for both task 1 and task2 can be found on the github repository inside the folder moodle_submissions
