#!bin/bash

echo -n "Who is the participant ?"
read participant
python create_original.py --participant $participant
python src/AVprocessing/camera.py
echo -n "Specify CAMERA ID"
read id
python main.py --id $id --participant $participant