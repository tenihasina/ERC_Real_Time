#!bin/bash
# Var assignment
LR=1e-4
EPOCHS = 60
MAG_BETA = 0.001

echo ========= lr=$LR ==============
for iter in 1 2 3 4 5 6 7 8 9 0
do
echo --- $Enc - $Dec $iter ---
python src/train_iemocap.py -lr $LR -epochs $EPOCHS -beta $MAG_BETA
done