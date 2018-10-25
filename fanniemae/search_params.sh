#!/bin/bash

TRAINING=./data/imb.train
VALIDATING=./data/imb.validate
TESTING=./data/imb.test

NUM_NODES=200
#NUM_LAYERS=('0' '1' '3' '5' '7' '9')
NUM_LAYERS=('5')

#rm config
#echo "HIDDEN_SIZE=$NUM_NODES" &> config
#echo "DEPTH=5" >> config

#python3.6 ./mortgage_test.py $TESTING model_5\_$NUM_NODES &> test_5\_$NUM_NODES.log

sleep 60

for n in "${NUM_LAYERS[@]}"
do
	rm config
	echo "HIDDEN_SIZE=$NUM_NODES" &> config
	echo "DEPTH=$n" >> config
	python3.6 ./mortgage_train.py $TRAINING $VALIDATING model_$n\_$NUM_NODES &> train_$n\_$NUM_NODES.log
	sleep 60
	python3.6 ./mortgage_test.py $TESTING model_$n\_$NUM_NODES &> test_$n\_$NUM_NODES.log
	sleep 60
done
