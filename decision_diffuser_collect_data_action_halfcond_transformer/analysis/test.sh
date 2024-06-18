#!/bin/bash

model_path=(1 3 4 5 6 3)
$model_index
$i

for ((i=0;i<6;i++))
do
    ((model_index=${model_path[$i]}))
    echo $model_index
done