#!/usr/bin/env bash
# todo: delete these when we make the repository public!

brats_2017_train="https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/2017/postChallenge/MICCAI_BraTS17_Data_Training_for_CBackes.zip"

brats_2017_validation="https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/2017/MICCAI_BraTS17_Data_Validation.zip"

brats_2018_train="https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/2018/MICCAI_BraTS_2018_Data_Training.zip"

wget "$brats_2017_train" &
wget "$brats_2017_validation" &
wget "$brats_2018_train" &