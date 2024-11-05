import numpy as np
import timeout_decorator
import traceback
from utils.logger import Logger
from utils.tf_utils import norm_grad
from collections import defaultdict
import json
import wandb
import os

from grad_compress.grad_drop import GDropUpdate
from grad_compress.sign_sgd import SignSGDUpdate, MajorityVote
from comm_effi import StructuredUpdate

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

from utils.tf_utils import cosine_sim, kl_divergence, normalize

L = Logger()
logger = L.get_logger()

DEVICE_MODELS = ['Google Nexus S', 'Xiaomi Redmi Note 7 Pro', 'Google Pixel 4']


class Server:

    def __init__(self, client_model, clients=[], cfg=None):
        self._cur_time = 0  # simulation time
        self.cfg = cfg
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.all_clients = clients
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []
        self.clients_info = defaultdict(dict)
        self.structure_updater = None
        self.update_failure = 0
        for c in self.all_clients:
            self.clients_info[str(c.id)]["comp"] = 0
            self.clients_info[str(c.id)]["acc"] = 0.0
            self.clients_info[str(c.id)]["device"] = c.device.device_model
            self.clients_info[str(c.id)]["sample_num"] = len(c.train_data['y'])

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        if num_clients < self.cfg.min_selected:
            logger.info('insufficient clients: need {} while get {} online'.format(self.cfg.min_selected, num_clients))
            return False
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, epoch=None, num_epochs=1, batch_size=10, minibatch=None, clients=None, deadline=-1):
        """Trains self.model on given clients.

        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            deadline: -1 for unlimited; >0 for each client's deadline
        Return:
            bytes_written: number of bytes written by each client to server
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0,
                   'acc': {},
                   'loss': {},
                   'ori_d_t': 0,
                   'ori_t_t': 0,
                   'ori_u_t': 0,
                   'act_d_t': 0,
                   'act_t_t': 0,
                   'act_u_t': 0} for c in clients}
        simulate_time = 0
        accs = []
        losses = []
        num_sumples_per_client = []
        low_end_devices_info = {
            'ori_d_ts': [],
            'ori_u_ts': [],
            'ori_t_ts': [],
            'act_d_ts': [],
            'act_u_ts': [],
            'act_t_ts': [],
        }
        moderate_devices_info = {
            'ori_d_ts': [],
            'ori_u_ts': [],
            'ori_t_ts': [],
            'act_d_ts': [],
            'act_u_ts': [],
            'act_t_ts': [],
        }
        high_end_devices_info = {
            'ori_d_ts': [],
            'ori_u_ts': [],
            'ori_t_ts': [],
            'act_d_ts': [],
            'act_u_ts': [],
            'act_t_ts': [],
        }
        succ_clients_info = {
            'device_models_occs': dict.fromkeys(DEVICE_MODELS, 0),
            'ori_d_ts': [],
            'ori_u_ts': [],
            'ori_t_ts': [],
            'act_d_ts': [],
            'act_u_ts': [],
            'act_t_ts': [],
            'all_bytes_written': [],
            'all_bytes_read': [],
            'all_local_comp': []
        }
        fail_clients_info = {
            'device_models_occs': dict.fromkeys(DEVICE_MODELS, 0),
            'ori_d_ts': [],
            'ori_u_ts': [],
            'ori_t_ts': [],
            'act_d_ts': [],
            'act_u_ts': [],
            'act_t_ts': [],
            'all_bytes_written': [],
            'all_bytes_read': [],
            'all_local_comp': []
        }
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []
        for c in clients:
            c.model.set_params(self.model)
            try:
                # set deadline
                c.set_deadline(deadline)
                # training
                logger.debug('client {} starts training...'.format(c.id))
                start_t = self.get_cur_time()
                # gradiant here is actually (-1) * grad
                simulate_time_c, comp, num_samples, update, acc, loss, gradiant, update_size, seed, shape_old, loss_old = c.train(
                    start_t, num_epochs, batch_size, minibatch)
                logger.debug('client {} simulate_time: {}'.format(c.id, simulate_time_c))
                logger.debug('client {} num_samples: {}'.format(c.id, num_samples))
                logger.debug('client {} acc: {}, loss: {}'.format(c.id, acc, loss))

                # Ahmed - Handle the case with NaN values - not working
                # if not np.isnan(acc):
                accs.append(acc)
                if not np.isnan(loss):
                    losses.append(loss)

                num_sumples_per_client.append(num_samples)
                simulate_time = min(deadline, max(simulate_time, simulate_time_c))
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += update_size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                sys_metrics[c.id]['acc'] = acc
                sys_metrics[c.id]['loss'] = loss
                # Update succ_clients_info dictionary
                succ_clients_info['device_models_occs'][c.get_device_model()] += 1
                succ_clients_info['all_bytes_written'].append(max(0.0, float(update_size)))
                succ_clients_info['all_bytes_read'].append(max(0.0, float(c.model.size)))
                succ_clients_info['all_local_comp'].append(max(0.0, float(comp)))
                # uploading
                self.updates.append((c.id, num_samples, update))

                if self.cfg.structure_k:
                    if not self.structure_updater:
                        self.structure_updater = StructuredUpdate(self.cfg.structure_k, seed)
                    gradiant = self.structure_updater.regain_grad(shape_old, gradiant)
                    # print("client {} finish using structure_update with k = {}".format(c.id, self.cfg.structure_k))

                self.gradiants.append((c.id, num_samples, gradiant))

                if self.cfg.qffl:
                    q = self.cfg.qffl_q
                    # gradiant = [(-1) * grad for grad in gradiant]
                    self.deltas.append([np.float_power(loss_old + 1e-10, q) * grad for grad in gradiant])
                    self.hs.append(q * np.float_power(loss_old + 1e-10, (q - 1)) * norm_grad(gradiant) + (
                                1.0 / self.client_model.lr) * np.float_power(loss_old + 1e-10, q))
                    # print("client {} finish using qffl with q = {}".format(c.id, self.cfg.qffl_q))

                norm_comp = int(comp / self.client_model.flops)
                if norm_comp == 0:
                    logger.error('comp: {}, flops: {}'.format(comp, self.client_model.flops))
                    assert False
                self.clients_info[str(c.id)]["comp"] += norm_comp
                # print('client {} upload successfully with acc {}, loss {}'.format(c.id,acc,loss))
                logger.debug('client {} upload successfully with acc {}, loss {}'.format(c.id, acc, loss))

            except timeout_decorator.timeout_decorator.TimeoutError as e:
                logger.debug('client {} failed: {}'.format(c.id, e))
                actual_comp = c.get_actual_comp()
                norm_comp = int(actual_comp / self.client_model.flops)
                self.clients_info[str(c.id)]["comp"] += norm_comp
                sys_metrics[c.id]['acc'] = -1
                sys_metrics[c.id]['loss'] = -1
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.update_size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] += actual_comp
                simulate_time = deadline
                # Update fail_clients_info dictionary
                fail_clients_info['device_models_occs'][c.get_device_model()] += 1
                fail_clients_info['all_bytes_written'].append(max(0.0, float(c.update_size)))
                fail_clients_info['all_bytes_read'].append(max(0.0, float(c.model.size)))
                fail_clients_info['all_local_comp'].append(max(0.0, float(actual_comp)))

            except Exception as e:
                logger.error('client {} failed: {}'.format(c.id, e))
                # logger.error('train_x: {}'.format(c.train_data['x']))
                # logger.error('train_y: {}'.format(c.train_data['y']))
                traceback.print_exc()
            finally:
                if self.cfg.compress_algo:
                    sys_metrics[c.id]['before_cprs_u_t'] = max(0.0, round(c.before_comp_upload_time, 3))
                sys_metrics[c.id]['ori_d_t'] = max(0.0, round(c.ori_download_time, 3))
                sys_metrics[c.id]['ori_t_t'] = max(0.0, round(c.ori_train_time, 3))
                sys_metrics[c.id]['ori_u_t'] = max(0.0, round(c.ori_upload_time, 3))

                sys_metrics[c.id]['act_d_t'] = max(0.0, round(c.act_download_time, 3))
                sys_metrics[c.id]['act_t_t'] = max(0.0, round(c.act_train_time, 3))
                sys_metrics[c.id]['act_u_t'] = max(0.0, round(c.act_upload_time, 3))

                if c.get_device_model() == 'Google Nexus S':
                    low_end_devices_info['ori_d_ts'].append(max(0.0, sys_metrics[c.id]['ori_d_t']))
                    low_end_devices_info['ori_t_ts'].append(max(0.0, sys_metrics[c.id]['ori_t_t']))
                    low_end_devices_info['ori_u_ts'].append(max(0.0, sys_metrics[c.id]['ori_u_t']))
                    low_end_devices_info['act_d_ts'].append(max(0.0, sys_metrics[c.id]['act_d_t']))
                    low_end_devices_info['act_t_ts'].append(max(0.0, sys_metrics[c.id]['act_t_t']))
                    low_end_devices_info['act_u_ts'].append(max(0.0, sys_metrics[c.id]['act_u_t']))
                elif c.get_device_model() == 'Xiaomi Redmi Note 7 Pro':
                    moderate_devices_info['ori_d_ts'].append(max(0.0, sys_metrics[c.id]['ori_d_t']))
                    moderate_devices_info['ori_t_ts'].append(max(0.0, sys_metrics[c.id]['ori_t_t']))
                    moderate_devices_info['ori_u_ts'].append(max(0.0, sys_metrics[c.id]['ori_u_t']))
                    moderate_devices_info['act_d_ts'].append(max(0.0, sys_metrics[c.id]['act_d_t']))
                    moderate_devices_info['act_t_ts'].append(max(0.0, sys_metrics[c.id]['act_t_t']))
                    moderate_devices_info['act_u_ts'].append(max(0.0, sys_metrics[c.id]['act_u_t']))
                elif c.get_device_model() == 'Google Pixel 4':
                    high_end_devices_info['ori_d_ts'].append(max(0.0, sys_metrics[c.id]['ori_d_t']))
                    high_end_devices_info['ori_t_ts'].append(max(0.0, sys_metrics[c.id]['ori_t_t']))
                    high_end_devices_info['ori_u_ts'].append(max(0.0, sys_metrics[c.id]['ori_u_t']))
                    high_end_devices_info['act_d_ts'].append(max(0.0, sys_metrics[c.id]['act_d_t']))
                    high_end_devices_info['act_t_ts'].append(max(0.0, sys_metrics[c.id]['act_t_t']))
                    high_end_devices_info['act_u_ts'].append(max(0.0, sys_metrics[c.id]['act_u_t']))

                if sys_metrics[c.id]['acc'] != -1:
                    succ_clients_info['ori_d_ts'].append(max(0.0, sys_metrics[c.id]['ori_d_t']))
                    succ_clients_info['ori_t_ts'].append(max(0.0, sys_metrics[c.id]['ori_t_t']))
                    succ_clients_info['ori_u_ts'].append(max(0.0, sys_metrics[c.id]['ori_u_t']))
                    succ_clients_info['act_d_ts'].append(max(0.0, sys_metrics[c.id]['act_d_t']))
                    succ_clients_info['act_t_ts'].append(max(0.0, sys_metrics[c.id]['act_t_t']))
                    succ_clients_info['act_u_ts'].append(max(0.0, sys_metrics[c.id]['act_u_t']))
                else:
                    fail_clients_info['ori_d_ts'].append(max(0.0, sys_metrics[c.id]['ori_d_t']))
                    fail_clients_info['ori_t_ts'].append(max(0.0, sys_metrics[c.id]['ori_t_t']))
                    fail_clients_info['ori_u_ts'].append(max(0.0, sys_metrics[c.id]['ori_u_t']))
                    fail_clients_info['act_d_ts'].append(max(0.0, sys_metrics[c.id]['act_d_t']))
                    fail_clients_info['act_t_ts'].append(max(0.0, sys_metrics[c.id]['act_t_t']))
                    fail_clients_info['act_u_ts'].append(max(0.0, sys_metrics[c.id]['act_u_t']))
        try:
            sys_metrics['configuration_time'] = simulate_time
            avg_train_acc = sum(accs) / len(accs)
            avg_train_loss = sum(losses) / len(losses)
            weighted_avg_train_acc = np.average(accs, weights=num_sumples_per_client, axis=0)
            weighted_avg_train_loss = np.average(losses, weights=num_sumples_per_client, axis=0)
            logger.info('Average Accuracy: {}, Average Loss: {}'.format(avg_train_acc, avg_train_loss))
            logger.info('Configuration and update stage simulation time: {}'.format(simulate_time))

            wandb.log({"Round/epoch": epoch,
                       "Round/Update failure": self.update_failure,
                       "Round/configuration time": simulate_time,
                       "Round/successful clients": len(self.updates),
                       "Round/failed clients": len(self.selected_clients) - len(self.updates),
                       "Average/Train Accuracy": avg_train_acc,
                       "Average/Train Loss": avg_train_loss,
                       "Weighted Average/Train Accuracy": weighted_avg_train_acc,
                       "Weighted Average/Train Loss": weighted_avg_train_loss,
                       "10th percentile/Train Accuracy": np.percentile(accs, 10, axis=0),
                       "50th percentile/Train Accuracy": np.percentile(accs, 50, axis=0),
                       "90th percentile/Train Accuracy": np.percentile(accs, 90, axis=0),
                       "10th percentile/Train Loss": np.percentile(losses, 10, axis=0),
                       "50th percentile/Train Loss": np.percentile(losses, 50, axis=0),
                       "90th percentile/Train Loss": np.percentile(losses, 90, axis=0),
                       "Variance/Train Accuracy": np.var(accs),
                       "Variance/Train Loss": np.var(losses)
                       }, step=epoch)

            # Ahmed - Fairness of clients train, upload and download time
            clients_upload = np.asarray([sys_metrics[c.id]['act_u_t'] for c in clients])
            clients_download =  np.asarray([sys_metrics[c.id]['act_d_t'] for c in clients])
            clients_train =  np.asarray([sys_metrics[c.id]['act_t_t'] for c in clients])
            wandb.log({"Jian Fairness/Actual upload time": (
                    1.0 / len(clients_upload) * (np.sum(clients_upload) ** 2) / np.sum(clients_upload ** 2)),
                "QoE Fairness/Actual upload time": (
                        1.0 - (2.0 * clients_upload.std() / (clients_upload.max() - clients_upload.min()))),
                "Jian Fairness/Actual download time": (
                        1.0 / len(clients_download) * (np.sum(clients_download) ** 2) / np.sum(clients_download ** 2)),
                "QoE Fairness/Actual download time": (
                        1.0 - (2.0 * clients_download.std() / (clients_download.max() - clients_download.min()))),
                "Jian Fairness/Actual train time": (
                        1.0 / len(clients_train) * (np.sum(clients_train) ** 2) / np.sum(clients_train ** 2)),
                "QoE Fairness/Actual train time": (
                        1.0 - (2.0 * clients_train.std() / (clients_train.max() - clients_train.min()))),
            }, step=epoch)


            if len(accs) > 0 and len(losses) > 0:
                train_acc = np.array(accs, dtype=np.float32)
                train_losses = np.array(losses, dtype=np.float32)
                wandb.log({
                    "Clients/Train Accuracy": wandb.Histogram(
                        np_histogram=np.histogram(train_acc, bins=10)),
                    "Clients/Train Loss": wandb.Histogram(
                        np_histogram=np.histogram(np.extract(np.isfinite(train_losses), train_losses), bins=10)),
                }, step=epoch)

                #Ahmed - Fairness of training accuracy
                wandb.log({"Jian Fairness/Train Accuracy": (
                            1.0 / len(train_acc) * (np.sum(train_acc) ** 2) / np.sum(train_acc ** 2)),
                           "QoE Fairness/Train Accuracy": (
                                       1.0 - (2.0 * train_acc.std() / (train_acc.max() - train_acc.min()))),
                           }, step=epoch)

            train_accs = np.asarray(accs)
            vectors_of_ones = np.ones(len(accs))
            wandb.log({
                "Cosine Similarity Metric/Train":
                    cosine_sim(train_accs, vectors_of_ones)
            }, step=epoch)

            uniform_vector = np.random.uniform(0, 1, len(accs))
            wandb.log({
                "KL Divergence Metric/Train":
                    kl_divergence(normalize(train_accs), uniform_vector)
            }, step=epoch)

            for device_model in DEVICE_MODELS:
                wandb.log({
                    "Successful Clients Info/" + "# " + device_model:
                        succ_clients_info['device_models_occs'][device_model],
                    "Fail Clients Info/" + "# " + device_model:
                        fail_clients_info['device_models_occs'][device_model]
                }, step=epoch)

            # Log clients' info
            local_comp_arr = np.asarray(succ_clients_info['all_local_comp'], dtype=np.float32)
            wandb.log({
                "Successful Clients Info/Local Computation": wandb.Histogram(
                    np_histogram=np.histogram(local_comp_arr, bins=10)),
                "Successful Clients Info/Bytes Written": wandb.Histogram(
                    np_histogram=np.histogram(np.asarray(succ_clients_info['all_bytes_written'], dtype=np.float32), bins=10)),
                "Successful Clients Info/Bytes Read": wandb.Histogram(
                    np_histogram=np.histogram(np.asarray(succ_clients_info['all_bytes_read'], dtype=np.float32), bins=10)),

                "Successful Clients Info/Average Original Download Time":
                    np.average(succ_clients_info['ori_d_ts'], axis=0),
                "Successful Clients Info/Average Original Upload Time":
                    np.average(succ_clients_info['ori_u_ts'], axis=0),
                "Successful Clients Info/Average Original Train Time":
                    np.average(succ_clients_info['ori_t_ts'], axis=0),
                "Successful Clients Info/Average Actual Download Time":
                    np.average(succ_clients_info['act_d_ts'], axis=0),
                "Successful Clients Info/Average Actual Upload Time":
                    np.average(succ_clients_info['act_u_ts'], axis=0),
                "Successful Clients Info/Average Actual Train Time":
                    np.average(succ_clients_info['act_t_ts'], axis=0),
                "Successful Clients Info/Average Bytes Written": np.average(succ_clients_info['all_bytes_written'],
                                                                            axis=0),
                "Successful Clients Info/Average Bytes Read": np.average(succ_clients_info['all_bytes_read'], axis=0),
                "Successful Clients Info/Average Local Computations": np.average(succ_clients_info['all_local_comp'],
                                                                                 axis=0),
            }, step=epoch)

            #Ahmed - Fairness of clients local computation
            wandb.log({"Jian Fairness/Success Local Computation": (
                        1.0 / len(local_comp_arr) * (np.sum(local_comp_arr) ** 2) / np.sum(local_comp_arr ** 2)),
                       "QoE Fairness/Success Local Computation": (
                                   1.0 - (2.0 * local_comp_arr.std() / (local_comp_arr.max() - local_comp_arr.min()))),
                       }, step=epoch)

            # Ahmed - Debug comp info for failed client - (IndexError: index -9223372036854775808 is out of bounds for axis 0 with size 11)
            logger.debug('failed comp: ' + str(fail_clients_info['all_local_comp']))
            wandb.log({
                "Fail Clients Info/Local Computation": wandb.Histogram(
                    np_histogram=np.histogram(np.asarray(fail_clients_info['all_local_comp'], dtype=np.float32), bins=10)),
                "Fail Clients Info/Bytes Written": wandb.Histogram(
                    np_histogram=np.histogram(np.asarray(fail_clients_info['all_bytes_written'], dtype=np.float32), bins=10)),
                "Fail Clients Info/Bytes Read": wandb.Histogram(
                    np_histogram=np.histogram(np.asarray(fail_clients_info['all_bytes_read'], dtype=np.float32), bins=10)),
                "Fail Clients Info/Average Original Download Time":
                    np.average(fail_clients_info['ori_d_ts'], axis=0),
                "Fail Clients Info/Average Original Upload Time":
                    np.average(fail_clients_info['ori_u_ts'], axis=0),
                "Fail Clients Info/Average Original Train Time":
                    np.average(fail_clients_info['ori_t_ts'], axis=0),
                "Fail Clients Info/Average Actual Download Time":
                    np.average(fail_clients_info['act_d_ts'], axis=0),
                "Fail Clients Info/Average Actual Upload Time":
                    np.average(fail_clients_info['act_u_ts'], axis=0),
                "Fail Clients Info/Average Actual Train Time":
                    np.average(fail_clients_info['act_t_ts'], axis=0),
                "Fail Clients Info/Average Bytes Written": np.average(fail_clients_info['all_bytes_written'], axis=0),
                "Fail Clients Info/Average Bytes Read": np.average(fail_clients_info['all_bytes_read'], axis=0),
                "Fail Clients Info/Average Local Computations": np.average(fail_clients_info['all_local_comp'], axis=0),
            }, step=epoch)

            # Log device models' info
            # TODO

        except ZeroDivisionError as e:
            logger.error('training time window is too short to train!')
            # assert False
        except Exception as e:
            logger.error('failed reason: {}'.format(e))
            traceback.print_exc()
            assert False
        return sys_metrics

    def update_model(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:
            logger.info('Round succeed, updating global model...')
            if self.cfg.no_training:
                logger.info('Pseduo-update because of no_training setting.')
                self.updates = []
                self.gradiants = []
                self.deltas = []
                self.hs = []
                return
            if self.cfg.aggregate_algorithm == 'FedAvg':
                # aggregate all the clients
                logger.info('Aggragate with FedAvg')
                used_client_ids = [cid for (cid, client_samples, client_model) in self.updates]
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_model) in self.updates:
                    total_weight += client_samples
                    for i, v in enumerate(client_model):
                        base[i] += (client_samples * v.astype(np.float64))
                for c in self.all_clients:
                    if c.id not in used_client_ids:
                        # c was not trained in this round
                        params = self.model
                        total_weight += c.num_train_samples  # assume that all train_data is used to update
                        for i, v in enumerate(params):
                            base[i] += (c.num_train_samples * v.astype(np.float64))
                averaged_soln = [v / total_weight for v in base]
                self.model = averaged_soln

            elif self.cfg.aggregate_algorithm == 'SucFedAvg':
                # aggregate the successfully uploaded clients
                logger.info('Aggragate with SucFedAvg')
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_model) in self.updates:
                    # logger.info('cid: {}, client_samples: {}, client_model: {}'.format(cid, client_samples, client_model[0][0][:5]))
                    total_weight += client_samples
                    for i, v in enumerate(client_model):
                        base[i] += (client_samples * v.astype(np.float64))
                averaged_soln = [v / total_weight for v in base]
                self.model = averaged_soln

            elif self.cfg.aggregate_algorithm == 'SelFedAvg':
                # aggregate the selected clients
                logger.info('Aggragate with SelFedAvg')
                used_client_ids = [cid for (cid, client_samples, client_model) in self.updates]
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_model) in self.updates:
                    total_weight += client_samples
                    for i, v in enumerate(client_model):
                        base[i] += (client_samples * v.astype(np.float64))
                for c in self.selected_clients:
                    if c.id not in used_client_ids:
                        # c was failed in this round but was selected
                        params = self.model
                        total_weight += c.num_train_samples  # assume that all train_data is used to update
                        for i, v in enumerate(params):
                            base[i] += (c.num_train_samples * v.astype(np.float64))
                averaged_soln = [v / total_weight for v in base]
                self.model = averaged_soln

            else:
                # not supported aggregating algorithm
                logger.error('not supported aggregating algorithm: {}'.format(self.cfg.aggregate_algorithm))
                assert False

        else:
            self.update_failure += 1
            logger.info('Round failed, global model maintained.')

        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []

    def update_using_compressed_grad(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.gradiants) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')
            if self.cfg.no_training:
                logger.info('pseduo-update because of no_training setting.')
                self.updates = []
                self.gradiants = []
                self.deltas = []
                self.hs = []
                return
            if self.cfg.aggregate_algorithm == 'FedAvg':
                # aggregate all the clients
                logger.info('Aggragate with FedAvg after grad compress')
                used_client_ids = [cid for (cid, _, _) in self.gradiants]
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_grad) in self.gradiants:
                    total_weight += client_samples
                    for i, v in enumerate(client_grad):
                        base[i] += (client_samples * v.astype(np.float32))
                for c in self.all_clients:
                    if c.id not in used_client_ids:
                        total_weight += c.num_train_samples
                averaged_grad = [v / total_weight for v in base]
                # update with grad
                if self.cfg.compress_algo == 'sign_sgd':
                    averaged_grad = MajorityVote(averaged_grad)
                self.model = self.client_model.update_with_gradiant(averaged_grad)

            elif self.cfg.aggregate_algorithm == 'SucFedAvg':
                # aggregate the successfully uploaded clients
                logger.info('Aggragate with SucFedAvg after grad compress')
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_grad) in self.gradiants:
                    # logger.info('cid: {}, client_samples: {}, client_model: {}'.format(cid, client_samples, client_model[0][0][:5]))
                    total_weight += client_samples
                    # logger.info("client-grad: {}, c-g[0]: {}".format(type(client_grad), type(client_grad[0])))
                    for i, v in enumerate(client_grad):
                        base[i] += (client_samples * v.astype(np.float32))
                averaged_grad = [v / total_weight for v in base]
                # update with grad
                if self.cfg.compress_algo == 'sign_sgd':
                    averaged_grad = MajorityVote(averaged_grad)
                self.model = self.client_model.update_with_gradiant(averaged_grad)

            elif self.cfg.aggregate_algorithm == 'SelFedAvg':
                # aggregate the selected clients
                logger.info('Aggragate with SelFedAvg after grad compress')
                used_client_ids = [cid for (cid, _, _) in self.gradiants]
                total_weight = 0.
                base = [0] * len(self.updates[0][2])
                for (cid, client_samples, client_grad) in self.gradiants:
                    total_weight += client_samples
                    for i, v in enumerate(client_grad):
                        base[i] += (client_samples * v.astype(np.float32))
                for c in self.selected_clients:
                    if c.id not in used_client_ids:
                        # c was failed in this round but was selected
                        total_weight += c.num_train_samples  # assume that all train_data is used to update
                averaged_grad = [v / total_weight for v in base]
                # update with grad
                if self.cfg.compress_algo == 'sign_sgd':
                    averaged_grad = MajorityVote(averaged_grad)
                self.model = self.client_model.update_with_gradiant(averaged_grad)

            else:
                # not supported aggregating algorithm
                logger.error('not supported aggregating algorithm: {}'.format(self.cfg.aggregate_algorithm))
                assert False

        else:
            self.update_failure += 1
            logger.info('round failed, global model maintained.')

        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []

    def update_using_qffl(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.gradiants) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')
            if self.cfg.no_training:
                logger.info('pseduo-update because of no_training setting.')
                self.updates = []
                self.gradiants = []
                self.deltas = []
                self.hs = []
                return

            # aggregate using q-ffl
            demominator = np.sum(np.asarray(self.hs))
            num_clients = len(self.deltas)
            scaled_deltas = []
            for client_delta in self.deltas:
                scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

            updates = []
            for i in range(len(self.deltas[0])):
                tmp = scaled_deltas[0][i]
                for j in range(1, len(self.deltas)):
                    tmp += scaled_deltas[j][i]
                updates.append(tmp)

            self.model = [(u - v) * 1.0 for u, v in zip(self.model, updates)]

            self.updates = []
            self.gradiants = []
            self.deltas = []
            self.hs = []

        else:
            self.update_failure += 1
            logger.info('Round failed, global model maintained.')

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients
            assert False

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            # logger.info('client {} metrics: {}'.format(client.id, c_metrics))
            metrics[client.id] = c_metrics
            if isinstance(c_metrics['accuracy'], np.ndarray):
                self.clients_info[client.id]['acc'] = c_metrics['accuracy'].tolist()
            else:
                self.clients_info[client.id]['acc'] = c_metrics['accuracy']

        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.all_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path, round):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess = self.client_model.sess
        return self.client_model.saver.save(model_sess, path, global_step=round)

    def restore_model(self, path):
        """Restore the server model on checkpoints/dataset/model.ckpt."""
        # restore server model
        #self.client_model.set_params(self.model)
        model_sess = self.client_model.sess
        return self.client_model.saver.restore(model_sess, path)

    def close_model(self):
        self.client_model.close()

    def get_cur_time(self):
        return self._cur_time

    def pass_time(self, sec):
        self._cur_time += sec

    def get_time_window(self):
        tw = np.random.normal(self.cfg.time_window[0], self.cfg.time_window[1])
        while tw < 0:
            tw = np.random.normal(self.cfg.time_window[0], self.cfg.time_window[1])
        return tw

    def save_clients_info(self):
        if not os.path.exists('logs/' + self.cfg.dataset):
            os.mkdir('logs/' + self.cfg.dataset)
        #with open(os.open('logs/{}/clients_info_{}.json'.format(self.cfg.dataset, self.cfg.config_name), os.O_CREAT | os.O_WRONLY, 0o774), 'w') as f:
        with open('logs/{}/clients_info_{}.json'.format(self.cfg.dataset, self.cfg.config_name), 'w') as f:
            json.dump(self.clients_info, f)
        logger.info('Save clients_info.json.')
