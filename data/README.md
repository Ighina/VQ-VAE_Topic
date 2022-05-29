# Use
Use this directory to store the data to feed your model.

# Structure
To use custom data, create a subfolder inside this directory. The subfolder should contain three additional subfolders named train, valid and test, each containing the training, validation and test text files, respectively. You can then pass your data to the run script by adding the data argument to the run script as follow: 
'''
python run.py --data data/<your_folder> *other arguments*
'''

For an example of how to format your data, see the subfolder named wiki50 in this directory.
