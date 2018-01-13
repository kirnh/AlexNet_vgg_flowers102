These are the steps to be taken in order to finetune AlexNet with the vgg_flowers102 dataset:

Step 1: Download data
Download the 102 class dataset and the labels matrix from http://www.robots.ox.ac.uk/~vgg/data/flowers/
and store it inside "data" directory so that the structure is;
|__data
    |__jpg
    |__imagelabels.mat

Also download the AlexNet pretrained weights from; 
https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy and place it in the root dir alongside
"data" dir.

If you directly want to do inference on the flower dataset, download already finetuned model files from 
the following link, place them inside "./_tmp/checkpoints/" and skip to step 4;
https://drive.google.com/open?id=1ZUjuJYee6f3Crg2G1Q1AXOcK6igMjpdQ 

Step 2: Prepare dataset
Create two empty text files called train.txt and val.txt inside "data" dir and run the 
restructure_data_alexnet.py script with the proper parameters set inside it that link to the files
created in step 1.

Step 3: Finetune AlexNet
Run finetune.py with the required parameters.

Step 4: Inference
After creating the model, set-up the inference script "infer.py" so that it can look up the required
model and image files. Then run it to see the predictions in terminal.
