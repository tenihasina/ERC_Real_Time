# ERC_Real_Time

* Use the yml file to create the virtual env 
* Might require to install portaudio and flac on windows to make the audio recording work

#KAMUH

This repository is a public implementation of the paper : "Knowledge Aware Multi-Headed Network with Dialogue Act and Emotion Shift Multi-Task Learning for Emotion Recognition in Conversation"

## Datasets

* DA annotations on MELD and IEMOCAP can be obtained at https://github.com/sahatulika15/EMOTyDA 
* DailyDialog : http://yanran.li/dailydialog
* IEMOCAP : fill request at https://sail.usc.edu/iemocap/
* MELD : https://affective-meld.github.io/

## Requirements

* CUDA >= 10.0
* Python >= 3.6
* Pytorch
* scikit-learn
* ATOMIC (extraction functions are already copied in repository) : https://github.com/atcbosselut/comet-commonsense

## Baseline

* Dialog-XL : https://github.com/shenwzh3/DialogXL
* AGHMN : https://github.com/wxjiao/AGHMN
* DialogueRNN : https://github.com/declare-lab/conv-emotion
* COSMIC : https://github.com/declare-lab/conv-emotion
* TODKAT : https://github.com/something678/TodKat

## Training & Evaluation

Train/Eval can be done using the .sh files and modifying the arguments inside if needed (default arguments corresponds to settings described in the article)