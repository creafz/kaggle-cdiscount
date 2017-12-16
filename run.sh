#!/bin/bash


CROP_LOCATIONS=(center top_left top_right bottom_left bottom_right)


python train.py \
    --model-name inceptionresnetv2 \
    --dropout-p 0.2 \
    --experiment-name cdiscount_inceptionresnetv2 \
    --batch-size 512 \
    --freeze-model \
    --seed 1 \
    --epochs 1;

python train.py \
    --model-name inceptionresnetv2 \
    --dropout-p 0.2 \
    --experiment-name cdiscount_inceptionresnetv2 \
    --batch-size 64 \
    --start-epoch 2 \
    --load-best-model \
    --seed 2 \
    --epochs 14;

python train.py \
    --model-name resnet152 \
    --experiment-name cdiscount_resnet152 \
    --img-size 160 160 \
    --batch-size 512 \
    --freeze-model \
    --seed 3 \
    --epochs 1;

python train.py \
    --model-name resnet152 \
    --experiment-name cdiscount_resnet152 \
    --img-size 160 160 \
    --batch-size 128 \
    --start-epoch 2 \
    --load-best-model \
    --seed 4 \
    --epochs 20;

for crop_location in "${CROP_LOCATIONS[@]}"
do
    python predict.py \
        --model-name resnet152 \
        --experiment-name cdiscount_resnet152 \
        --crop \
        --crop-location ${crop_location};
done

python predict.py \
    --model-name inceptionresnetv2 \
    --experiment-name cdiscount_inceptionresnetv2 \
    --dropout-p 0.2;

python predict.py \
    --model-name inceptionresnetv2 \
    --experiment-name cdiscount_inceptionresnetv2 \
    --dropout-p 0.2 \
    --flip-lr;

python predict.py \
    --model-name inceptionresnetv2 \
    --experiment-name cdiscount_inceptionresnetv2 \
    --dropout-p 0.2 \
    --flip-ud;

python predict.py \
    --model-name inceptionresnetv2 \
    --experiment-name cdiscount_inceptionresnetv2 \
    --dropout-p 0.2 \
    --flip-lr \
    --flip-ud;

python merge_predictions.py \
    cdiscount_resnet152/npz_crop_160x160_top_left/ \
    cdiscount_resnet152/npz_crop_160x160_top_right/ \
    cdiscount_resnet152/npz_crop_160x160_center/ \
    cdiscount_resnet152/npz_crop_160x160_bottom_left/ \
    cdiscount_resnet152/npz_crop_160x160_bottom_right/ \
    --target-dir cdiscount_resnet152/merged;

python merge_predictions.py \
    cdiscount_inceptionresnetv2/npz/ \
    cdiscount_inceptionresnetv2/npz_fliplr/ \
    cdiscount_inceptionresnetv2/npz_flipud/ \
    cdiscount_inceptionresnetv2/npz_fliplr_flipud/ \
    --target-dir cdiscount_inceptionresnetv2/merged;

python merge_predictions.py \
    cdiscount_resnet152/merged/ \
    cdiscount_inceptionresnetv2/merged/ \
    --target-dir submission_predictions \
    --averaging gmean;

python make_submission.py \
    --directory submission_predictions \
    --submission-filename submission.csv;
