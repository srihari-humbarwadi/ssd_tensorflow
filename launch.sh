for file in cfg/*
do
screen -dmS $(basename $file .yaml)  python3 train.py $file
done
