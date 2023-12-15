#!bin/bash

#echo -n "Who is the participant ?"
#read participant
#echo -n "Which mic ?"
#read id_micro
#python create_original.py --participant $participant --id_micro $id_micro
python src/AVprocessing/camera.py
#echo -n "Specify CAMERA ID"
#read id
/home/sandratra/anaconda3/envs/thesis/bin/python3 /home/sandratra/Documents/thesis/ERC_Real_Time/video.py
#sudo /home/sandratra/anaconda3/envs/thesis/bin/python3 /home/sandratra/Documents/thesis/ERC_Real_Time/main.py