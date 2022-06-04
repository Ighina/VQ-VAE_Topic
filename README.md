# VQ-VAE_Topic
An implementation of the paper [Vector-Quantization-Based Topic Modeling](https://dl.acm.org/doi/10.1145/3450946), providing a series of VQ-VAE models for topic modelling. The model reaches state-of-the-art performance on Ng20 and enables the extraction of dense topic vectors for downstream tasks.

For a quick overview, consult the google colab file VQ_VAE_Program.ipynb in this project.

# Installation
Clone the repository and then install the dependecies by running
```
pip install -r requirements.txt
```
in your terminal or virtual environment.
## Download Word Embeddings
To run the program you will first need to download GloVe word embeddings, unzip them and copy the relative files in the data directory. You can download the version of GloVe we used from [here](https://nlp.stanford.edu/data/glove.6B.zip). Once downloaded the zipped folder, copy it to the data directory and unzip it: the default version of GloVe embeddings used by run.py is the one contained in "data/glove.6B.300d.txt", but you can change it by passing the --glove_file option to the run script and defining the new location after it:
```
python run.py --glove_file data/<your preferred GloVe file>
```
# Usage
## Basic Usage
The script can be run immediately by using
```
python run.py
```
This will train the Hard VQ-VAE model on cpu over the preprocessed Ng20 dataset used in the original paper (and store in the data directory). If you want to run the same program on a single gpu, instead, run the following (multiple gpus have not been tested yet):
```
python run.py -gpu 1
```
If you are comparing the results with other topic models, it is **strongly advised** to add the --evaluate_on_target option, as it will force the coherence evaluation to consider just words present in the training set, rather than all the available GloVe vocabulary (thus matching the configuration of the majority of topic models). The basic usage with such a setting is as follow:
```
python run.py --evaluate_on_target
```
or
```
python run.py -eot
```
## Saving Outputs
To save the topic vectors and topic-word matrix resulting from your experiment, add the --save_topic argument as follow:
```
python run.py --save_topic
```
or
```
python run.py -st
```
Both topic vectors and topic-words matrix are saved by default in the experiments folder, under the name "topic_vecs<experiment number>.npy" and "topic_matrix<experiment_number>.csv", respectively; <experiment number> is the number of experiments run and it is automatically computed as the number of files already present in the experiment folder divided by 3 (that is the number of the outputs of a program when including --save_topic).
## Using different VQ-VAE Models
Following the orginal paper, we included three different VQ-VAE models, plus a variation not included in the original work.
### Hard VQ-VAE
This is the default model, where each word embedding is associated with just a single topic vector in the forward pass. The basic usage is as before.
### Soft VQ-VAE
This model create a soft association between words and topic vectors in the forward pass, by means of an attention mechanism. Use this setting as follow:
```
python run.py --soft
```
or
```
python run.py -s
```
### Multi-View
This is the implementation of the original multi-view architecture in the paper, that substitutes the simple attention mechanism of Soft VQ-VAE with something similar to a multi-head attention. Use this option as follow:
```
python run.py -s --multi_view
```
or
```
python run.py -s -mv
```
You can specify the number of views to be used as:
```
python run.py -s --multi_view --heads <desired number of views>
```
or
```
python run.py -s -mv -nh
```
### Multi-Head
We also included a classic multi-head attention that can be used instead of the authors' suggested multi-view architecture. The difference is that with multi-head there are more parameters and we have empirically observed that multi-head tends to work better. Again, the use is:
```
python run.py -s --multi_head
```
or
```
python run.py -s -mh <desired number of views>
```
Also here, you can specify the number of heads to be used as:
```
python run.py -s --multi_view --heads <desired number of heads>
```
or
```
python run.py -s -mh -nh <desired number of heads>
```
## Saved Models
Once a training script is succesfully completed, you will find the saved model as a torch checkpoint in the pretrained-models directory. You might want to rename the model with a different name for further use.

# Other Options
 Many other options including training parameters and preprocessing options can be passed to the run.py script. To see all the available options, run the script as follow:
```
python run.py --help
```
# Using your data
 To use your data as input to the model, follow the instructions in the data folder to import your data in the correct structure and to run the script.

 
# Predicting probabilities with pre-trained model
 Once you have trained a model, you can use it to predict probabilities and topic vectors for an input corpus. 
 After training, the run.py script automatically saves hyperparameters and model checkpoint in the folder you specified.
 Use the predict.py script by passing your data location and the hyerparameters.json stored in the pretrained model folder as follow:
 ```
python predict.py --data inputs/<your data folder> --hyperparameters experiments/<pre-trained model folder>/hyperparameters.json --out_directory outputs/<a name for your output directory>
```
 or
  ```
python predict.py -d inputs/<your data folder> --hp experiments/<pre-trained model folder>/hyperparameters.json -out outputs/<a name for your output directory>
```
 
 You can optionally also save the document vectors generated by the model (by averaging the relevant topic vectors): to do so add the --return_topiv_vectors or -rtv parameter to the above scripts.
 
 For an example of how your input data for prediction should be structured look at the inputs/CNN10 folder in this project.
 
 Probabilities and document vectors are stored in the output directory specified inside numpy files (.npy)
