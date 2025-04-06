# Deep-Learning-Sound-Classification

### Setup ###
Dataset is from: https://commonvoice.mozilla.org/en/datasets

Run 
$ tar -xzvf cv-corpus-12.0-delta-2022-12-07-en.tar.gz -C data/ 
on the zip file to extract the data (do it in the project root file)


### CNN ###
Before training the model, you need run:
python generate_spectograms.py
This will generate the .npy image spectograms of each sample in the processed_spectograms/ directory

Then you can run the following commands for each model

    ## CNN ##
    python main.py --task gender --model cnn
    and
    python main.py --task age --model cnn

    ## MLP ##
    python main.py --task gender --model mlp
    and
    python main.py --task age --model mlp

    ## GCNN ##
    python main.py --task gender --model gcnn
    and
    python main.py --task age --model gcnn
