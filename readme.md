# NJIT677 - Deep Learning
## Project 1

### Ted Moore, David Apolinar, Shawn Cicoria

## HOW TO RUN
Call src/baseline.py from the command line:

`python ./src/baseline.py`

This will run a new-classes scenario on the the Core50 dataset, with the pretrained ResNet18 model by default. Parameters defined below.

### The script assumes you have the following directory structure:

1) A folder, named `core50/` at the same level as `src/`, which contains the default downloaded output of Continuum's Core50 data class. 
`Core50("/core50/", train=True, download=True)`

2) A file, `core50_train.csv` which includes the image paths of the data you wish to use for training the algorithm. The rest of the images will be witheld for validation.

### Available CLI arguments:

> '-cls', '--classifier', type=str, default='resnet18', choices=['resnet18', 'resnet101']

> '--lr', type=float, default=0.00001, help='learning rate'

> '--batch_size', type=int, default=32, help='batch_size'

> '--epochs', type=int, default=4, help='number of epochs')

> '--weight_decay', type=float, default=0.000001, help='weight decay'

> '--convergence_criterion', type=float, default=0.004, help='convergence_criterion ')

> '--momentum', type=float, default=0.8, help='momentum'

> '--replay', type=float, default=0.15, help='proportion of training examples to replay'

Of course, the script is made publicly available, so further modifications (e.g. different classifiers, or a New Instances scenario) can easily be implemented.

### On Continuous Learning

The currently implemented approach to continuous learning is in the form of Rehearsal, where a proportion of the previous tasks' training examples are selected at random and appended to the current task training examples. This can theoretically help a model not "forget" previous examples, by interspersing them with the current task. This approach has its shortcomings, including additional training time and required memory, but can be useful in conjunction with other methods.
