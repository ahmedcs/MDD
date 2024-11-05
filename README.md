# Model Discovery and Distillation (MDD)

- An Federated Learning implementation which is based on FLASH framework designed to Simulate *Heterogeneity-Aware* Federated Learning
- an updated version of FLASH can be found: [HERE](https://github.com/PKU-Chengxu/FLASH)


## How to run it

### Installing

#### Pip packages
```bash
git clone https://github.com/ahmedcs/mdd.git
pip3 install -r requirements.txt
# download data, modify code if needed, refer to Chapter.Dataset for more details
```
#### Conda environment
```bash
git clone https://github.com/ahmedcs/mdd.git
cd ibex
bash create-conda-env.sh
# download data, modify code if needed, refer to Chapter.Dataset for more details
```

### results in paper

For details on the experimental results, please refer to our paper.

The experimental configs are stored a folder for each data set in `models/exp_config`.

To run your own experiment, you can just modify the `models/exp_config/${dataset}/default.cfg` and then run `python main.py --config ${path_to_config}`.

## What is MDD?

Briefly speaking, we develop MDD to experiment with Model Discovery and Distillation within the federated learning simulation environment.

### Deadline

We add deadline setting for simulating failed downloading/uploading and time out training. Now deadline follows a normal distribution in each round, and each client has the same deadline in one round. You can set the deadline's normal distribution parameters in the config file.



### Heterogeneity

#### Hardware Heterogeneity

Each client is bundled with a device type. Each device type has different training speeds and network speeds. We also support self-defined device type(-1) whose parameter you can set in the code manually for more complexed simulation. Note that if a client's device is not specified i.e. None, the program will use real training time instead of the simulation time, which is not recommended.

The source code for measure the on-device training time is available in the [android](android/) directory



#### Behavior Heterogeneity

Each client is bundled with a timer, which is bundled with one trace. Timer gets the available time according to [google definition](https://arxiv.org/pdf/1902.01046.pdf). FLASH will run in ideal mode if trace file is not found or `behav_hete` is set to `False`



#### Data Heterogeneity

- data in each client is non-i.i.d
- you can set `max_sample` to control the max sample number in each client


### Round Failure

In federated settings, if there are not enough devices to upload the results in a round, then this round will be regarded as a failed round and the global model will not be updated. To simulate it, we add a update_frac parameter. If the uploaded fraction is smaller than update_frac, then this round will fail. You can also set it in the config file.



### Config

To simplify the command line arguments, we move most of the parameters to a config file. Also, we add some other parameters as put above for better simulation. Here are some details.

```bash
# line started with # (commented) will be ignored
behav_hete True
# bool, whether to simulate behavior heterogeneity
hard_hete True
# bool, whether to simulate hardware heterogeneity, which contains differential on-device training time and network speed
no_training False
# bool, whether to run in no_training mode, skip training process if True
real_world False
# bool, whether to run read-world DL dataset
dataset femnist
# dataset to use
model cnn
# file that defines the DNN model
num_rounds 500
# number of FL rounds to run
learning_rate 0.01
# learning-rate of DNN
eval_every 5
# evaluate every # rounds, -1 for not evaluate
clients_per_round 100
# expected clients in each round
min_selected 60
# min selected clients number in each round, fail if not satisfied
max_sample 340
# number of max sampleto use in each selected client
batch_size 10
# batch-size for training
num_epochs 5
# number epochs imn each client in each round
seed 0
# basic random seed
round_ddl 270 0
# μ and σ for deadline, which follows a normal distribution
update_frac 0.8  
# min update fraction in each round, round succeeds only when fraction of succeeded client not less than #
max_client_num -1
#

# NOTE! [aggregate_algorithm, fedprox*, structure_k, qffl*] is mutually-exclusive
aggregate_algorithm SucFedAvg
## choose in [SucFedAvg, FedAvg], please refer to models/server.py for more details
# compress_algo grad_drop
## gradiant compress algorithm, choose in [grad_drop, sign_sgd], not use if commented
fedprox True
fedprox_mu 0.5
fedprox_active_frac 0.8
## whether to apply fedprox and params needed, please refer to the sysml'20 for more details
# structure_k 100
## the k for structured update, not use if commented, please refer to the arxiv for more
# qffl True
# qffl_q 5
## whether to apply qffl and params needed, please refer to the ICLR'20 for more
```

## Datasets

### DL Datasets

#### FEMNIST

- **Overview:** Image Dataset
- **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
- **Task:** Image Classification

#### Celeba

- **Overview:** Image Dataset based on the [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Details:** 9343 users (we exclude celebrities with less than 5 images)
- **Task:** Image Classification (Smiling vs. Not smiling)

#### Reddit

- **Overview:** We preprocess the Reddit data released by [pushshift.io](https://files.pushshift.io/reddit/) corresponding to December 2017.
- **Details:** 1,660,820 users with a total of 56,587,343 comments.
- **Task:** Next-word Prediction.

### User Behavior Trace

- you can download the user behavior trace data [here](https://drive.google.com/file/d/1M62hlt2KpF6IRGZi_gEDv7DE3_s0_ihD/view?usp=sharing).

- modify the file path in [models/client.py](models/client.py),

  i.e.  `with open('/path/to/user_behavior_trace.json', 'r', encoding='utf-8') as f: `

- The trace tracks the device’s meta information and its status changes, including battery charge status, battery level, network environment, screen lock status, and screen on and off. (See more details in our [manuscript]().)


## On-device Training

the code we used to measure the on-device training time is in `OnDeviceTraining` directory

please refer to the [doc](OnDeviceTraining/README.md) for more details

## Commands used

Running the experiment on IBEX
```
bash ibex/submit_exp.sh 10:59:59 exp_config/uncompleted_runs shakespeare "" 0
```
Generating the missing runs from WANDB
```
python regenerate_uncompleted_runs.py 0 490 17750 femnist celeba shakespeare reddit sent140
```


## Notes

- Install the libraries listed in ```requirements.txt```

  - i.e. with pip: run ```pip3 install -r requirements.txt```

- Go to directory of respective dataset `data/$DATASET` for instructions on generating data

- please consider to cite our paper if you use the code or data in your research project

  ```
  ```
