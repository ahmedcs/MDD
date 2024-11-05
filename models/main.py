"""Script to run the baselines."""
import argparse
import eventlet
import importlib
import numpy as np
import pandas as pd
import os
import sys
import random
import time
import signal
import json
import traceback
import tensorflow as tf
import wandb
import math
import copy

from utils.tf_utils import norm_grad


mask = 0o771
os.umask(0) #create the files with group permissions
#sys.setrecursionlimit(10000)

from collections import defaultdict, Counter

from utils.tf_utils import cosine_sim, kl_divergence, normalize

import metrics.writer as metrics_writer

from exp_config.regenerate_runs import get_runs

# args
from utils.args import parse_args
eventlet.monkey_patch()
args = parse_args()
config_file  = args.config
config_name = '.'.join(config_file.split('/')[-1].split('.')[:-1])

from utils.config import Config
# Read config from file
cfg = Config(config_file)

cfg.sample = args.sample
cfg.resume = args.resume
# if cfg.sample == "":
#     train_data_dir = os.path.join('..', 'data', cfg.dataset, 'data', 'train')
#     filenames = os.listdir(train_data_dir)
#     if len(filenames) > 0 and filenames[0].find('_niid_') < 0:
#         cfg.sample = "iid"
#     else:
#         cfg.sample = "niid"

# logger
from utils.logger import Logger
L = Logger()
L.set_log_name(cfg.config_name, cfg.dataset)
logger = L.get_logger()

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from utils.model_utils import read_data
from device import Device

DEVICE_MODELS = ['Google Nexus S', 'Xiaomi Redmi Note 7 Pro', 'Google Pixel 4']

root_path = args.root_dir #".."

def main():

    # Set metrics dir path
    args.metrics_dir = 'metrics/' + cfg.dataset + '/' + cfg.config_name + '/'
    os.makedirs(args.metrics_dir, exist_ok=True)

    # Initiate wandb
    #train_data_dir = os.path.join('..', 'data', cfg.dataset, 'data', 'train')
    #filenames = os.listdir(train_data_dir)
    # if len(filenames) > 0 and filenames[0].find('_niid_') < 0 and (not cfg.dataset.find('niid')):
    #     run = wandb.init(project='flash_' + cfg.dataset + '_iid', entity='fed_bias', name=cfg.config_name,
    #                      config=dict(cfg.__dict__), resume=cfg.dataset + '_' + cfg.config_name)
    # else:
    #     run = wandb.init(project='flash_' + cfg.dataset, entity='fed_bias', name=cfg.config_name,
    #                      config=dict(cfg.__dict__), resume=cfg.dataset + '_' + cfg.config_name)

    #Determine if we use folder name or file name for iid and niid
    foldername=True
    if cfg.sample == "":
        train_data_dir = os.path.join(root_path, 'data', cfg.dataset, 'data', 'train')
        filenames = os.listdir(train_data_dir)
        if len(filenames) > 0 and filenames[0].find('_niid_') >= 0:
            cfg.sample = "niid"
        elif len(filenames) > 0 and filenames[0].find('_iid_') >= 0:
            cfg.sample = "iid"
        foldername=False

    # # exit if the job has completed or currently running
    # uncompleted, complete, tagged = get_runs(cfg.sample, cfg.dataset, cfg.seed, cfg.num_rounds)
    # if cfg.config_name in complete:
    #     logger.info('------------------ Previously Finished or Running Experiment - Exiting --------------------')
    #     exit(0)

    #Init Wandb
    if cfg.sample == "":
        project = cfg.dataset
    else:
        project = cfg.dataset + '_' + cfg.sample
    if cfg.resume:
        run = wandb.init(project='flash_' + project, entity='ahmedcs982', name=cfg.config_name,
                         config=dict(cfg.__dict__), id=cfg.dataset + '_' + cfg.config_name, resume=cfg.resume)
    else:
        run = wandb.init(project='flash_' + project, entity='ahmedcs982', name=cfg.config_name,
                         config=dict(cfg.__dict__))


    # cfg.sample = ""
    # if len(cfg.dataset.split('_')) > 1:
    #     cfg.dataset = cfg.dataset.split('_')[0]
    #     cfg.sample = cfg.dataset.split('_')[1]
    #wandb.config.update(cfg.__dict__)
    #logger.info(run.config)
    #wandb.config.setdefaults(cfg.__dict__)

    #log configuration
    cfg.log_config()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + cfg.seed)
    np.random.seed(12 + cfg.seed)
    tf.compat.v1.set_random_seed(123 + cfg.seed)

    model_path = '%s/%s.py' % (cfg.dataset, cfg.model)
    if not os.path.exists(model_path):
        logger.error('Please specify a valid dataset and a valid model.')
        assert False
    model_path = '%s.%s' % (cfg.dataset, cfg.model)

    logger.info('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    '''
    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]
    '''

    num_rounds = cfg.num_rounds
    eval_every = cfg.eval_every
    clients_per_round = cfg.clients_per_round

    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if cfg.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = cfg.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(cfg.seed, *model_params, cfg=cfg)
    logger.info('Model size: {}'.format(client_model.size))

    # Create clients
    logger.info('======================Setup Clients==========================')
    clients = setup_clients(cfg, model=client_model, dirname=foldername)

    label_occurences = []
    #Ahmed - initially add all train labels to the frequency array
    # if cfg.dataset != 'reddit':
    #     label_occurences = [Counter(c.train_data["y"] for c in clients)]
    # else:
    #     label_occurences = []
    #     for c in clients:
    #         for item in c.train_data["y"]:
    #             if item is None:
    #                 continue
    #             temp = item['target_tokens']
    #             for tokens in temp:
    #                 label_occurences.append(Counter(' '.join(tokens)))
    #logger.info('client labels: ' + str(label_occurences))
    #train_samples_num = sorted([c.num_train_samples for c in clients])
    #logger.info(train_samples_num)

    attended_clients = set()

    # Create server
    server = Server(client_model, clients=clients, cfg=cfg)

    resume_round = 0
    # ckpt_path = os.path.join('checkpoints', cfg.dataset + '_' + cfg.sample)
    # if cfg.resume and os.path.exists(ckpt_path):
    #     # Restore server model
    #     # round_file = os.path.join(ckpt_path, '{}_{}.txt'.format(cfg.model, cfg.config_name))
    #     # save_path = os.path.join(ckpt_path, '{}_{}.ckpt'.format(cfg.model, cfg.config_name))
    #     rounds = []
    #     files = {}
    #     for file in os.listdir(ckpt_path):
    #         if os.path.isfile(os.path.join(ckpt_path, file)) and '.ckpt' in file and cfg.config_name in file:
    #             if file.find('-') >= 0:
    #                 val = int(file.split('/')[-1].split('.')[1].split('-')[-1])
    #                 rounds.append(val)
    #                 files[val]= file
    #             else:
    #                 files[0] = file
    #     #for file in files:
    #     if len(files) > 0:
    #         #resume_round = int(file.split('-')[1].split('.')[0]) + 1
    #         resume_round = max(files.keys())
    #         file = files[resume_round]
    #         chk_path = os.path.join(ckpt_path, file.split('.')[0] + '.' + file.split('.')[1])
    #         server.restore_model(chk_path)
    #         logger.info('Model restored in path: %s at round %s' % (chk_path, resume_round))

    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)

    # Initial status
    logger.info('===================== Random Initialization =====================')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)

    # Simulate training
    if num_rounds == -1:
        import sys
        num_rounds = sys.maxsize

    def timeout_handler(signum, frame):
        raise Exception

    def exit_handler(signum, frame):
        os._exit(0)

    c_ids_num_selected = dict.fromkeys(client_ids, 0)
    c_ids_num_succeded = dict.fromkeys(client_ids, 0)
    c_ids_num_failed = dict.fromkeys(client_ids, 0)

    #count of the success, fail, selection
    sel_count = 0
    suc_count = 0
    fail_count = 0

    selection_failure = 0

    # Ahmed - KDN train randomly selected slow clients using the initial model
    #low_end = []
    #for c in clients:
        #if c.get_device_model() == 'Google Nexus S':
            #low_end.append(c)
            #c.model.lr *= cfg.clients_per_round * .8
    #rand_low_end = random.sample(low_end, 10)
    rand_low_end = random.sample(clients, 10)
    for x in rand_low_end:
        clients.remove(x)

    # Ahmed ---- KDN copy initial model
    initial_model = copy.deepcopy(server.client_model.get_params())
    acc = []
    for x in rand_low_end:
        val = x.model.test(x.eval_data)['accuracy']
        acc.append(val)
    logger.info('START model accuracy avg {} list {}'.format(np.average(acc), acc))

    for i in range(resume_round, num_rounds):
        # round_start_time = time.time()
        logger.info('===================== Round {} of {} ====================='.format(i+1, num_rounds))

        # 1. selection stage
        logger.info('--------------------- Selection Stage ---------------------')
        # 1.1 select clients
        cur_time = server.get_cur_time()
        time_window = server.get_time_window()
        logger.info('Current time: {}\ttime window: {}\t'.format(cur_time, time_window))
        online_clients = online(clients, cur_time, time_window)
        if not server.select_clients(i,
                              online_clients,
                              num_clients=clients_per_round):
            # insufficient clients to select, round failed
            selection_failure += 1
            logger.info('Round failed in selection stage!')
            server.pass_time(time_window)
            continue
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)
        attended_clients.update(c_ids)
        c_ids.sort()
        logger.info("Selected num: {}".format(len(c_ids)))
        logger.debug("Selected client_ids: {}".format(c_ids))

        successful_clients = []

        for c_id in c_ids:
            c_ids_num_selected[c_id] += 1

        # 1.2 decide deadline for each client
        deadline = np.random.normal(cfg.round_ddl[0], cfg.round_ddl[1])
        while deadline <= 0:
            deadline = np.random.normal(cfg.round_ddl[0], cfg.round_ddl[1])
        deadline = int(deadline)
        #Ahmed - enforce deadline for device heter as well
        if cfg.behav_hete: #or cfg.hard_hete:
            logger.info('Selected deadline: {} - cfg deadline: {}'.format(deadline, cfg.round_ddl[0]))

        # 1.3 update simulation time
        server.pass_time(time_window)

        # # Ahmed ---- KDN copy initial model
        # if i == resume_round:
        #     initial_model = copy.deepcopy(server.client_model.get_params())
        #     acc = []
        #     for x in rand_low_end:
        #         val = x.model.test(x.eval_data)['accuracy']
        #         acc.append(val)
        #     logger.info('START model accuracy avg {} list {}'.format(np.average(acc), acc))

        # 2. configuration stage
        logger.info('--------------------- Configuration Stage ---------------------')
        # 2.1 train(no parallel implementation)
        sys_metrics = server.train_model(epoch=i, num_epochs=cfg.num_epochs, batch_size=cfg.batch_size, minibatch=cfg.minibatch, deadline = deadline)
        sys_writer_fn(i, c_ids, sys_metrics, c_groups, c_num_samples)

        for c_id in c_ids:
            if sys_metrics[c_id]['acc'] != -1:
                c_ids_num_succeded[c_id] += 1
                successful_clients.append(c_id)
            else:
                c_ids_num_failed[c_id] += 1

        for client in server.selected_clients:
            if client.id in successful_clients:
                if cfg.dataset == 'reddit':
                    for item in client.train_data["y"]:
                        if item is None:
                            continue
                        temp = item['target_tokens']
                        for tokens in temp:
                            # logger.info(client.id, ''.join(tokens))
                            label_occurences.append(Counter(''.join(tokens)))
                else: #if cfg.dataset == 'femnist' or cfg.dataset == 'celeba' or cfg.dataset == 'sent140' or cfg.dataset == 'shakespeare':
                    #logger.info(client.id, client.train_data["y"])
                    label_occurences.append(Counter(client.train_data["y"]))

        # 2.2 update simulation time
        server.pass_time(sys_metrics['configuration_time'])

        # 3. update stage
        logger.info('--------------------- Report Stage ---------------------')
        # 3.1 update global model
        if cfg.compress_algo:
            logger.info('Update using compressed grads')
            server.update_using_compressed_grad(cfg.update_frac)
        elif cfg.qffl:
            server.update_using_qffl(cfg.update_frac)
            logger.info('Round success by using qffl')
        else:
            server.update_model(cfg.update_frac)

        #log info to wandb
        # Ahmed - log the number of attended and test clients
        wandb.log({'Round/attended clients': len(attended_clients),
                   'Round/online clients': len(online_clients), 'Round/selected clients': len(c_ids),
                   'Round/time window': time_window, 'Round/simulation time': cur_time,
                   'Round/deadline': deadline, 'Round/selection failure': selection_failure}
                  , step=i)

        # Ahmed - log the percentage of success and failed clients
        sel_count += len(c_ids)
        suc_count += len(successful_clients)
        fail_count += (len(c_ids) - len(successful_clients))
        wandb.log({'Round/aggregate failed': 1.0 * fail_count / sel_count,
                   'Round/aggregate success': 1.0 * suc_count / sel_count
                   }, step=i)

        # 4. Test model(if necessary)
        # if eval_every == -1:
        #     continue

        if (eval_every != -1 and (i + 1) % eval_every == 0) or (i + 1) == num_rounds:
            if cfg.no_training:
                continue
            logger.info('--------------------- test result ---------------------')
            logger.info('Attended_clients num: {}/{}'.format(len(attended_clients), len(clients)))

            #Ahmed - choose a reasonable number of clients for Sent140 due to its size
            #problem is clients in sent140 is too many
            test_num = min(len(clients), 3500)

            test_clients = random.sample(clients, test_num)
            sc_ids, sc_groups, sc_num_samples = server.get_clients_info(test_clients)
            logger.info('Number of clients for test: {} of {} '.format(len(test_clients),len(clients)))
            another_stat_writer_fn = get_stat_writer_function(sc_ids, sc_groups, sc_num_samples, args)
            # print_stats(i + 1, server, test_clients, client_num_samples, args, stat_writer_fn)
            print_stats(i, server, test_clients, sc_num_samples, args, another_stat_writer_fn)

            if (i + 1) % (10 * eval_every) == 0 or (i + 1) == num_rounds:
                # Log clients info DataFrame as well
                clients_info_df = pd.DataFrame()
                for client in clients:
                    info = {'ID': client.id,
                            '#Selected': c_ids_num_selected[client.id],
                            '#Succeded': c_ids_num_succeded[client.id],
                            '#Failed': c_ids_num_failed[client.id],
                           }
                    clients_info_df = clients_info_df.append(info, ignore_index=True)

                wandb.log({"Clients Info DataFrame":
                            wandb.Table(columns=clients_info_df.columns.tolist(), #['#Failed', '#Succeded', '#Selected', 'ID'],
                                        data=clients_info_df.values.tolist())
                          }, step=i)

                # Ahmed - Fairness of the client selection
                clients_selected = np.asarray(list(c_ids_num_selected.values()))
                wandb.log({"Jian Fairness/Client selection": (
                        1.0 / len(clients_selected) * (np.sum(clients_selected) ** 2) / np.sum(clients_selected ** 2)),
                    "QoE Fairness/Client selection": (
                            1.0 - (2.0 * clients_selected.std() / (clients_selected.max() - clients_selected.min()))),
                }, step=i)

                clients_success = np.asarray(list(c_ids_num_succeded.values()))
                wandb.log({"Jian Fairness/Client Success": (
                        1.0 / len(clients_success) * (np.sum(clients_success) ** 2) / np.sum(clients_success ** 2)),
                    "QoE Fairness/Client Success": (
                            1.0 - (2.0 * clients_success.std() / (clients_success.max() - clients_success.min()))),
                }, step=i)

                clients_fail = np.asarray(list(c_ids_num_failed.values()))
                wandb.log({"Jian Fairness/Client Fail": (
                        1.0 / len(clients_fail) * (np.sum(clients_fail) ** 2) / np.sum(clients_fail ** 2)),
                    "QoE Fairness/Client Fail": (
                            1.0 - (2.0 * clients_fail.std() / (clients_fail.max() - clients_fail.min()))),
                }, step=i)

                # Count of each class of data (label) to be trained on
                total_label_occurences = Counter()
                for lo in label_occurences:
                    total_label_occurences.update(lo)

                #logger.info('number of labels:', str(total_label_occurences), str(label_occurences))
                if len(total_label_occurences) > 0:
                    # Ahmed - Frequency and Fairness of label occurance
                    total_label_occurences_df = pd.DataFrame.from_dict(total_label_occurences, orient='index').reset_index()
                    label_occurences_arr = np.asarray([total_label_occurences[x] for x in total_label_occurences])
                    wandb.log({"Labels Aggregate Count DataFrame":
                                   wandb.Table(columns=['Label', 'Aggregate Count'],
                                               # total_label_occurences_df.columns.tolist(),#
                                               data=total_label_occurences_df.values.tolist()),
                               "Jian Fairness/label occurance": (
                                       1.0 / len(label_occurences_arr) * (np.sum(label_occurences_arr) ** 2) / np.sum(
                                   label_occurences_arr ** 2)),
                               "QoE Fairness/label occurance": (
                                       1.0 - (2.0 * label_occurences_arr.std() / (
                                       label_occurences_arr.max() - label_occurences_arr.min()))),

                               }, step=i)

                #Save clients information
                server.save_clients_info()

                # Save server model
                if cfg.sample != "":
                    ckpt_path = os.path.join('checkpoints', cfg.dataset + "_" + cfg.sample)
                else:
                    ckpt_path = os.path.join('checkpoints', cfg.dataset)
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                for f in os.listdir(ckpt_path):
                    str = '{}_{}'.format(cfg.model, cfg.config_name)
                    if str in f:
                        logger.debug('removing old checkpoint: ' + f)
                        os.remove(os.path.join(ckpt_path, f))
                # f = open(os.path.join(ckpt_path, '{}_{}.txt'.format(cfg.model, cfg.config_name)), mode='w')
                # f.write(str(i))
                # f.close()
                #save_path = server.save_model(os.path.join(ckpt_path, '{}_{}.ckpt'.format(cfg.model, cfg.config_name)), i)
                #logger.info('Model saved in path: %s at round %s' % (save_path,i))

    #-------------------------------------------------------------------------------
    # Ahmed - KDN save the last version of the model

    final_model = copy.deepcopy(server.client_model.get_params())
    logger.info("FL model initial norm %f server norm %f copy norm %f", norm_grad(initial_model), norm_grad(server.client_model.get_params()), norm_grad(final_model))

    fl_acc = []
    for x in rand_low_end:
        val = server.client_model.test(x.eval_data)['accuracy']
        fl_acc.append(val)
    logger.info('END model update fail {} accuracy avg {} list {}'.format(server.update_failure, np.average(fl_acc), fl_acc))

    #Ahmed - KDN Train the models on initial model
    base_models = []
    before_base_acc = []
    base_acc = []
    base_avg_acc = 0
    before_base_avg_acc = 0
    count_arr = []

    # old accuracy
    c_models = []
    #acc = []
    for i, c in enumerate(rand_low_end):
        c_models.append(copy.deepcopy(initial_model))
        c.model.set_params(c_models[i])
        logger.info("BEFORE BASE id:%s client norm %f server norm %f", c.id, norm_grad(c.model.get_params()), norm_grad(final_model) )
        val = c.model.test(c.eval_data)['accuracy']
        #acc.append(val)
        #before_base_avg_acc += val
        before_base_acc.append(val)

    epoch_vals = [5, 45, 50]
    base_acc = []
    for k, val in enumerate(epoch_vals):
        for i, c in enumerate(rand_low_end):
            avg_loss = []
            count = 0
            for epochs in range(0, val):
                comp, update, grad, acc, loss, loss_old, num_train_samples = c.train_local()
                avg_loss.append(loss)
                count += 1
            c_models[i] = copy.deepcopy(c.model.get_params())
            # while abs(loss - np.average(avg_loss)) / np.average(avg_loss) >= 0.025:
            #     comp, update, grad, acc, loss, loss_old, num_train_samples = c.train_local()
            #     avg_loss.append(loss)
            #     count += 1
            #count_arr.append(count)
            #base_models.append(update)

        #new accuracy
        temp_acc = []
        for i, c in enumerate(rand_low_end):
            c.model.set_params(c_models[i])
            logger.info("AFTER BASE id:%s client norm %f server norm %f", c.id, norm_grad(c.model.get_params()), norm_grad(final_model) )
            val = c.model.test(c.eval_data)['accuracy']
            #acc.append(val)
            #base_avg_acc += val
            temp_acc.append(val)
        base_acc.append(temp_acc)

    #base_avg_acc /= 100
    #before_base_avg_acc /= 100
    logger.info('train count: {}'.format(count_arr))
    logger.info('BASE BEFORE - avg {} list {}'.format(np.average(before_base_acc), before_base_acc))
    logger.info('BASE 5 AFTER - avg {} list {}'.format(np.average(base_acc[0]), base_acc[0]))
    logger.info('BASE 50 AFTER - avg {} list {}'.format(np.average(base_acc[1]), base_acc[1]))
    logger.info('BASE 100 AFTER - avg {} list {}'.format(np.average(base_acc[2]), base_acc[1]))

    # Ahmed - KDN Train the models on final model
    kdn_models = []
    before_kdn_acc = []
    kdn_acc = []
    kdn_avg_acc = 0
    before_kdn_avg_acc = 0
    count_arr = []

    # old new accuracy
    #acc = []
    for i, c in enumerate(rand_low_end):
        c.model.set_params(initial_model)
        logger.info("BEFORE KDN id:%s client norm %f server norm %f", c.id, norm_grad(c.model.get_params()), norm_grad(final_model) )
        val = c.model.test(c.eval_data)['accuracy']
        #acc.append(val)
        #before_kdn_avg_acc += val
        before_kdn_acc.append(val)

    for i, c in enumerate(rand_low_end):
        c.model.set_params(final_model)

        # Ahmed - KDN average the models
        #c.model.set_params(c_models[i])
        #c.average_with_model(final_model)
        #c_models[i] = copy.deepcopy(c.model.get_params())

        # count = 0
        for epochs in range(0, 100):
            comp, update, grad, acc, loss, loss_old, num_train_samples = c.train_local()
        c_models[i] = copy.deepcopy(c.model.get_params())
        #      avg_loss.append(loss)
        #      count += 1
        # # while abs(loss - np.average(avg_loss)) / np.average(avg_loss) >= 0.025:
        # #     comp, update, grad, acc, loss, loss_old, num_train_samples = c.train_local()
        # #     avg_loss.append(loss)
        # #     count += 1
        # count_arr.append(count)
        # kdn_models.append(update)

    #new accuracy
    #acc = []
    for i, c in enumerate(rand_low_end):
        c.model.set_params(c_models[i])
        logger.info("AFTER KDN id:%s client norm %f server norm %f", c.id, norm_grad(c.model.get_params()), norm_grad(final_model))
        val = c.model.test(c.eval_data)['accuracy']
        #acc.append(val)
        #kdn_avg_acc += val
        kdn_acc.append(val)

    #before_kdn_avg_acc /= 100
    #kdn_avg_acc /= 100
    #Ahmed - KDN log the acc values to wandb and print
    logger.info('train count: {}'.format(count_arr))
    logger.info('KDN BEFORE - avg {} list {}'.format(np.average(before_kdn_acc), before_kdn_acc))
    logger.info('KDN AFTER - avg {} list {}'.format(np.average(kdn_acc), kdn_acc))

    diff_acc = []
    for k, a in enumerate(base_acc):
        b = kdn_acc[k]
        diff_acc.append(np.subtract(b, a))

    # wandb.log({'before_base_avg_acc': before_base_avg_acc,
    #            'before_kdn_avg_acc': base_avg_acc,
    #            'base_avg_acc': before_kdn_avg_acc,
    #            'kdn_avg_acc': kdn_avg_acc,
    #            'diff_avg_acc': kdn_avg_acc-base_avg_acc},
    #           step=i)
    #
    # wandb.log({'before_base_models_acc': wandb.Histogram(np_histogram=np.histogram(np.concatenate(before_base_acc).flat,bins=10)),
    #            'before_kdn_models_acc': wandb.Histogram(np_histogram=np.histogram(np.concatenate(before_kdn_acc).flat, bins=10)),
    #             'base_models_acc': wandb.Histogram(np_histogram=np.histogram(np.concatenate(base_acc).flat, bins=10)),
    #            'kdn_models_acc': wandb.Histogram(np_histogram=np.histogram(np.concatenate(kdn_acc).flat, bins=10)),
    #            'diff_models_acc': wandb.Histogram(np_histogram=np.histogram(np.concatenate(diff_acc).flat, bins=10))},
    #           step=i)

    outfile = open(cfg.dataset+".csv", 'w')
    kdn_norouting_acc1 = base_acc[2].copy()
    for i in range(0, int(0.1 * len(kdn_norouting_acc1))):
        index = random.randint(0, len(kdn_norouting_acc1) - 1)
        kdn_norouting_acc1[index] = kdn_acc[index]

    np.savetxt(outfile, kdn_norouting_acc1, delimiter=",")
    np.savetxt(cfg.dataset + '_norouting_kdn_0.1.csv', kdn_norouting_acc1, delimiter=",")

    kdn_norouting_acc2 = base_acc[2].copy()
    for i in range(0, int(0.3 * len(kdn_norouting_acc2))):
        index = random.randint(0, len(kdn_norouting_acc2)-1)
        kdn_norouting_acc2[index] = kdn_acc[index]
    np.savetxt(outfile, kdn_norouting_acc2,  delimiter=",")
    np.savetxt(cfg.dataset + '_norouting_kdn_0.3.csv', kdn_norouting_acc2,  delimiter=",")

    kdn_norouting_acc3 = base_acc[2].copy()
    for i in range(0, int(0.5 * len(kdn_norouting_acc3))):
        index = random.randint(0, len(kdn_norouting_acc3) - 1)
        kdn_norouting_acc3[index] = kdn_acc[index]
    np.savetxt(outfile, kdn_norouting_acc3, delimiter=",")
    np.savetxt(cfg.dataset + '_norouting_kdn_0.5.csv', kdn_norouting_acc3, delimiter=",")

    kdn_norouting_acc4 = base_acc[2].copy()
    for i in range(0, int(0.7 * len(kdn_norouting_acc4))):
        index = random.randint(0, len(kdn_norouting_acc4) - 1)
        kdn_norouting_acc4[index] = kdn_acc[index]
    np.savetxt(outfile, kdn_norouting_acc4, delimiter=",")
    np.savetxt(cfg.dataset + '_norouting_kdn_0.7.csv', kdn_norouting_acc4, delimiter=",")

    np.savetxt(outfile, kdn_acc,  delimiter=",")
    np.savetxt(cfg.dataset + '_kdn.csv', kdn_acc,  delimiter=",")
    np.savetxt(outfile, fl_acc,  delimiter=",")
    np.savetxt(cfg.dataset + '_fl.csv', fl_acc,  delimiter=",")
    #np.savetxt(outfile, diff_acc,  delimiter=",")
    np.savetxt(cfg.dataset + '_diff.csv', diff_acc,  delimiter=",")

    np.savetxt(outfile,before_base_acc, delimiter=",")
    np.savetxt(cfg.dataset + '_base.csv',before_base_acc, delimiter=",")
    np.savetxt(outfile, base_acc[0], delimiter=",")
    np.savetxt(cfg.dataset + '_base_5.csv', base_acc[0], delimiter=",")
    np.savetxt(outfile, base_acc[1], delimiter=",")
    np.savetxt(cfg.dataset + '_base_50.csv', base_acc[1], delimiter=",")
    np.savetxt(outfile, base_acc[2], delimiter=",")
    np.savetxt(cfg.dataset + '_base_100.csv', base_acc[2], delimiter=",")

    averages = [np.average(kdn_norouting_acc1), np.average(kdn_norouting_acc2), np.average(kdn_norouting_acc3), np.average(kdn_norouting_acc4),
                np.average(kdn_acc),np.average(fl_acc), np.average(before_base_acc),
                np.average(base_acc[0]), np.average(base_acc[1]), np.average(base_acc[2])]

    np.savetxt(cfg.dataset + '_averages.csv', averages, delimiter=",")

    outfile.close()

    # Close models
    server.close_model()

def online(clients, cur_time, time_window):
    # """We assume all users are always online."""
    # return online client according to client's timer
    online_clients = []
    for c in clients:
        try:
            if c.timer.ready(cur_time, time_window):
                online_clients.append(c)
        except Exception as e:
            traceback.print_exc()
    L = Logger()
    logger = L.get_logger()
    logger.info('{} of {} clients online'.format(len(online_clients), len(clients)))
    return online_clients


def create_clients(users, groups, train_data, test_data, model, cfg):
    L = Logger()
    logger = L.get_logger()
    client_num = min(cfg.max_client_num, len(users))
    users = random.sample(users, client_num)
    logger.info('Clients in Total: %d' % (len(users)))
    if len(groups) == 0:
        groups = [[] for _ in users]

    all_device_types = []
    if cfg.low_end_dev_frac != -1 and cfg.moderate_dev_frac != -1 and cfg.high_end_dev_frac != -1:
        low_end_devices = int(np.round(cfg.low_end_dev_frac * len(users)))
        moderate_devices = int(np.round(cfg.moderate_dev_frac * len(users)))
        high_end_devices = int(np.round(cfg.high_end_dev_frac * len(users)))

        #Fix the issue with remaining unassigned client
        total = low_end_devices + moderate_devices + high_end_devices
        moderate_devices += client_num - total

        all_device_types.extend(['Google Nexus S'] * low_end_devices)
        all_device_types.extend(['Xiaomi Redmi Note 7 Pro'] * moderate_devices)
        all_device_types.extend(['Google Pixel 4'] * high_end_devices)

        random.shuffle(all_device_types)
    else:
        all_device_types.extend([None] * len(users))

    cnt = 0
    clients = []
    for u, g in zip(users, groups):
        c = Client(u, g, train_data[u], test_data[u], model, Device(cfg, device_model=all_device_types.pop(), model_size=model.size), cfg)
        if len(c.train_data["x"]) == 0:
            continue
        clients.append(c)
        cnt += 1
        if cnt % 500 == 0:
            logger.info('Set up {} clients'.format(cnt))

    from timer import Timer
    #Timer.save_cache()
    model2cnt = defaultdict(int)
    for c in clients:
        model2cnt[c.get_device_model()] += 1
    logger.info('Device setup result: {}'.format(model2cnt))

    return clients


def setup_clients(cfg, model=None, use_val_set=False, dirname=True):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    if cfg.sample != "" and dirname:
        train_data_dir = os.path.join(root_path, 'data', cfg.dataset + '_' + cfg.sample, 'data', 'train')
        test_data_dir = os.path.join(root_path, 'data', cfg.dataset + '_' + cfg.sample, 'data', eval_set)
    else:
        train_data_dir = os.path.join(root_path, 'data', cfg.dataset, 'data', 'train')
        test_data_dir = os.path.join(root_path, 'data', cfg.dataset, 'data', eval_set)

    swap = False
    if os.path.exists(train_data_dir):
        filenames = os.listdir(train_data_dir)
        if len(filenames) > 0 and ((filenames[0].find('_niid_') >= 0 and cfg.sample != 'niid') or (
                filenames[0].find('_iid_') >= 0 and cfg.sample != 'iid')):
            swap = True

    if swap or not os.path.exists(train_data_dir):
        if dirname:
            train_data_dir = os.path.join(root_path, 'data', cfg.dataset, 'data', 'train')
            test_data_dir = os.path.join(root_path, 'data', cfg.dataset, 'data', eval_set)
        else:
            train_data_dir = os.path.join(root_path, 'data', cfg.dataset + '_' + cfg.sample, 'data', 'train')
            test_data_dir = os.path.join(root_path, 'data', cfg.dataset + '_' + cfg.sample, 'data', eval_set)


    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    clients = create_clients(users, groups, train_data, test_data, model, cfg)

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        #Ahmed - stop writing the metrics
        #metrics_writer.print_metrics(
        #    num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))
        pass
    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        #Ahmed - stop writting the metrcis
        #metrics_writer.print_metrics(
        #    num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))
        pass
    return writer_fn


def print_stats(num_round, server, clients, num_samples, args, writer):

    # train_stat_metrics = server.test_model(clients, set_to_use='train')
    # print_metrics(train_stat_metrics, num_samples, prefix='train_')
    # writer(num_round, train_stat_metrics, 'train')

    test_stat_metrics = server.test_model(clients, set_to_use='test')
    print_metrics(num_round, test_stat_metrics, num_samples, prefix='test_')
    writer(num_round, test_stat_metrics, 'test')


def print_metrics(num_round, metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    client_ids = [c for c in sorted(metrics.keys())]
    num_clients = len(client_ids)
    ordered_weights = [weights[c] for c in client_ids]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    L = Logger()
    logger = L.get_logger()
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in client_ids]
        logger.info('{}: {}, 10th percentile: {}, 50th percentile: {}, 90th percentile {}'.format
                (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights, axis=0),
                 np.percentile(ordered_metric, 10, axis=0),
                 np.percentile(ordered_metric, 50, axis=0),
                 np.percentile(ordered_metric, 90, axis=0)))

        if metric == 'accuracy':
            test_accs = np.asarray(ordered_metric)

            wandb.log({"Average/Test Accuracy": np.average(test_accs, axis=0),
                       "Weighted Average/Test Accuracy": np.average(test_accs, weights=ordered_weights, axis=0),
                       "10th percentile/Test Accuracy": np.percentile(test_accs, 10, axis=0),
                       "50th percentile/Test Accuracy": np.percentile(test_accs, 50, axis=0),
                       "90th percentile/Test Accuracy": np.percentile(test_accs, 90, axis=0),
                       "Variance/Test Accuracy": np.var(test_accs),
                       "Round/test clients": len(test_accs),
                       }, step=num_round)

            if len(test_accs) > 0:
                wandb.log({"Clients/Test Accuracy": wandb.Histogram(
                           np_histogram=np.histogram(test_accs, bins=10)),
                       }, step=num_round)

                # Ahmed - Fairness of test accuracy
                wandb.log({"Jian Fairness/Test Accuracy": (1.0/len(test_accs) * (np.sum(test_accs)**2) / np.sum(test_accs**2)),
                           "QoE Fairness/Test Accuracy": (1.0 - (2.0 * test_accs.std() / (test_accs.max() - test_accs.min()))),
                           }, step=num_round)


            # Compute an log cosine similarity metric with input vector a of
            # clients' accuracies and b the same-length vector of 1s
            vectors_of_ones = np.ones(num_clients)
            wandb.log({
                "Cosine Similarity Metric/Test":
                    cosine_sim(test_accs, vectors_of_ones)
            }, step=num_round)

            # Compute an log KL Divergence metric with input vector a of
            # clients' normalized accuracies and the vector b of same-length
            # generated from the uniform distribution
            uniform_vector = np.random.uniform(0, 1, num_clients)
            wandb.log({
                "KL Divergence Metric/Test":
                    kl_divergence(normalize(test_accs), uniform_vector)
            }, step=num_round)

        elif metric == 'loss':
            test_losses = np.asarray(ordered_metric, dtype=np.float32)
            #logger.info(test_losses)
            wandb.log({"Average/Test Loss": np.average(test_losses, axis=0),
                       "Weighted Average/Test Loss": np.average(test_losses, weights=ordered_weights, axis=0),
                       "10th percentile/Test Loss": np.percentile(test_losses, 10, axis=0),
                       "50th percentile/Test Loss": np.percentile(test_losses, 50, axis=0),
                       "90th percentile/Test Loss": np.percentile(test_losses, 90, axis=0),
                       "Variance/Test Loss": np.var(test_losses)
                       }, step=num_round)

            if len(test_losses) > 0:
                wandb.log({"Clients/Test Loss": wandb.Histogram(
                           np_histogram=np.histogram(np.extract(np.isfinite(test_losses), test_losses), bins=10)),
                       }, step=num_round)

if __name__ == '__main__':
    # nohup python main.py -dataset shakespeare -model stacked_lstm &
    start_time=time.time()
    tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
    main()
    # logger.info("used time = {}s".format(time.time() - start_time))
