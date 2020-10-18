# NJIT677 - Deep Learning
## Project 1

### Ted Moore, David Apolinar, Shawn Cicoria

## Setup
For this solution Anaconda was used.  You need to install and then run an environment setup using:

```bash
conda env create -n <YOUR ENVIRONMENT NAME> --file ./environment.yaml
```


## HOW TO RUN
Call src/baseline.py from the command line:

`python ./src/baseline.py --data_path ./data --download True`

This will run a new-classes scenario on the the Core50 dataset, with the pretrained ResNet18 model by default. Parameters defined below.

### The script creates the data path

1) The above command creates a local path `./data` that contains the downloaded `core50_128x128.zip` file **VERY LARGE** and then expands the file to the `./data/core50_128x128` path. The full size of the zip and extracted contents is **12+ Gigabytes**

>Note: You must have enough disk space to accomodate the full core50 data set that is downloaded and extracted -- **12+ Gigabytes**

2) A file, `core50_train.csv` which includes the image paths of the data you wish to use for training the algorithm. The rest of the images will be witheld for validation.

### Available CLI arguments:

```
usage: ./src/baseline.py  [-h] [--data_path DATA_PATH]
                              [--download DOWNLOAD]
                              [-cls {resnet18,resnet101,resnet34}] [--lr LR]
                              [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                              [--weight_decay WEIGHT_DECAY]
                              [--convergence_criterion CONVERGENCE_CRITERION]
                              [--momentum MOMENTUM] [--replay REPLAY]
                              [--importance IMPORTANCE]
                              [--use_parallel USE_PARALLEL]
                              [--outfile OUTFILE]
```

#### Some of the settings and their defaults.

> '-cls', '--classifier', type=str, default='resnet18', choices=['resnet18', 'resnet101']

> '--lr', type=float, default=0.00001, help='learning rate'

> '--batch_size', type=int, default=32, help='batch_size'

> '--epochs', type=int, default=4, help='number of epochs')

> '--weight_decay', type=float, default=0.000001, help='weight decay'

> '--convergence_criterion', type=float, default=0.004, help='convergence_criterion ')

> '--momentum', type=float, default=0.8, help='momentum'

> '--replay', type=float, default=0.15, help='proportion of training examples to replay'

Of course, the script is made publicly available, so further modifications (e.g. different classifiers, or a New Instances scenario) can easily be implemented -- pull requests are welcome.

### On Continuous Learning

The currently implemented approach to continuous learning is in the form of Rehearsal, where a proportion of the previous tasks' training examples are selected at random and appended to the current task training examples. This can theoretically help a model not "forget" previous examples, by interspersing them with the current task. This approach has its shortcomings, including additional training time and required memory, but can be useful in conjunction with other methods.
