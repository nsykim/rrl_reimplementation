import torch
import torch.nn as nn
from collections import defaultdict
from sklearn import metrics
import logging
import sys
import numpy as np

from src.layers import *

TEST_CNT_MOD = 500

class Net(nn.Module):
    def __init__(self, dim_list, use_not=False, left=None, right=None, use_nlaf=False, estimated_grad=False, use_skip=True, alpha=0.999, beta=8, gamma=1, temperature=0.01):
        super(Net, self).__init__()
        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])
        self.use_skip = use_skip
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

        self._initialize_layers(dim_list, use_nlaf, estimated_grad, alpha, beta, gamma)

    def _initialize_layers(self, dim_list, use_nlaf, estimated_grad, alpha, beta, gamma):
        print(f"dim_list: {dim_list}")
        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            skip_from_layer = None
            if self.use_skip and i >= 4:
                skip_from_layer = self.layer_list[-2]
                num += skip_from_layer.output_size

            if i == 1:
                layer = FeatureBinarizer(dim_list[i], num, use_negation=self.use_not, min_val=self.left, max_val=self.right)
                layer_name = f'binary{i}'
            elif i == len(dim_list) - 1:
                layer = LinearRegressionLayer(dim_list[i], num)
                layer_name = f'lr{i}'
            else:
                layer_use_not = i != 2
                layer = UnionLayer(dim_list[i], num, use_novel_activation=use_nlaf, estimated_grad=estimated_grad, use_negation=layer_use_not, alpha=alpha, beta=beta, gamma=gamma)
                layer_name = f'union{i}'

            self._set_layer_connections(layer, skip_from_layer)
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def _set_layer_connections(self, layer, skip_from_layer):
        layer.conn = lambda: None
        layer.conn.prev_layer = self.layer_list[-1] if len(self.layer_list) > 0 else None
        layer.conn.is_skip_to_layer = False
        layer.conn.skip_from_layer = skip_from_layer
        if skip_from_layer is not None:
            skip_from_layer.conn.is_skip_to_layer = True

    def forward(self, x):
        print(f"X: {x.shape}")
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                print(f"layer is in conn.skip_fron_layer idk X: {x.shape}")
                del layer.conn.skip_from_layer.x_res
            x = layer(x) # calls the forward method of the layer
            print(f"X after layer {layer}: {x.shape}")
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
        print(f"X after all layers: {x.shape}")
        return x
    
    def binarized_forward(self, x, count=False):
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res
            x = layer.binarized_forward(x)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
            if count and layer.layer_type != 'linear':
                layer.node_activation_cnt += torch.sum(x, dim=0)
                layer.forward_tot += x.shape[0]
        return x


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list
    
    @property
    def t(self):
        return self.module.t


class RRL:
    def __init__(self, dim_list, device_id, use_not=False, is_rank0=False, log_file=None, writer=None, left=None,
                 right=None, save_best=False, estimated_grad=False, save_path=None, distributed=True, use_skip=False, 
                 use_nlaf=False, alpha=0.999, beta=8, gamma=1, temperature=0.01):
        self.dim_list = dim_list
        self.use_not = use_not
        self.use_skip = use_skip
        self.use_nlaf = use_nlaf
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.best_f1 = -1.
        self.best_loss = 1e20

        self.device_id = device_id
        self.is_rank0 = is_rank0
        self.save_best = save_best
        self.estimated_grad = estimated_grad
        self.save_path = save_path
        self.writer = writer

        self._setup_logging(log_file)
        self.net = self._initialize_net(dim_list, use_not, left, right, use_nlaf, estimated_grad, use_skip, alpha, beta, gamma, temperature, distributed)

    def _setup_logging(self, log_file):
        if self.is_rank0:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
            if log_file is None:
                logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
            else:
                logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)

    def _initialize_net(self, dim_list, use_not, left, right, use_nlaf, estimated_grad, use_skip, alpha, beta, gamma, temperature, distributed):
        net = Net(dim_list, use_not=use_not, left=left, right=right, use_nlaf=use_nlaf, estimated_grad=estimated_grad, use_skip=use_skip, alpha=alpha, beta=beta, gamma=gamma, temperature=temperature)
        net.cuda(self.device_id)
        if distributed:
            net = MyDistributedDataParallel(net, device_ids=[self.device_id])
        return net

    def clip(self):
        for layer in self.net.layer_list[:-1]:
            layer.clip_weights()
    
    def edge_penalty(self):
        edge_penalty = 0.0
        for layer in self.net.layer_list[1:-1]:
            edge_penalty += layer.edge_count()
        return edge_penalty
        # return sum(layer.edge_count() for layer in self.net.layer_list[1:-1])

    
    def l1_penalty(self):
        l1_penalty = 0.0
        for layer in self.net.layer_list[1:-1]:
            l1_penalty += layer.l1_norm()
        return l1_penalty
        # return sum(layer.l1_norm() for layer in self.net.layer_list[1:])
    
    def l2_penalty(self):
        l2_penalty = 0.0
        for layer in self.net.layer_list[1:-1]:
            l2_penalty += layer.l2_norm()
        return l2_penalty
        # return sum(layer.l2_norm() for layer in self.net.layer_list[1:])
    
    def mixed_penalty(self):
        penalty = 0.0
        for layer in self.net.layer_list[1:-1]:
            penalty += layer.l2_norm()
        penalty+= self.net.layer_list[-1].l1_norm()
        return penalty
        # penalty = sum(layer.l2_norm() for layer in self.net.layer_list[1:-1])
        # penalty += self.net.layer_list[-1].l1_norm()
        # return penalty

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, data_loader=None, valid_loader=None, epoch=50, lr=0.01, lr_decay_epoch=100, 
                    lr_decay_rate=0.75, weight_decay=0.0, log_iter=50):

        if data_loader is None:
            raise Exception("Data loader is unavailable!")

        accuracy_b = []
        f1_score_b = []

        criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.0)

        cnt = -1
        avg_batch_loss_rrl = 0.0
        epoch_histc = defaultdict(list)
        for epo in range(epoch):
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch)

            epoch_loss_rrl = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0

            ba_cnt = 0
            for X, y in data_loader:
                ba_cnt += 1
                X = X.cuda(self.device_id, non_blocking=True)
                y = y.cuda(self.device_id, non_blocking=True)
                optimizer.zero_grad()
                
                print(f"X: {X.shape}, y: {y.shape}")
                y_bar = self.net.forward(X) / torch.exp(self.net.t)
                y_arg = torch.argmax(y, dim=1)
                print(f"y_bar: {y_bar.shape} with value: {y_bar}")
                print(f"y_arg: {y_arg.shape} with value: {y_arg}")
                
                loss_rrl = criterion(y_bar, y_arg) + weight_decay * self.l2_penalty()
                
                ba_loss_rrl = loss_rrl.item()
                epoch_loss_rrl += ba_loss_rrl
                avg_batch_loss_rrl += ba_loss_rrl
                
                loss_rrl.backward()

                cnt += 1
                with torch.no_grad():
                    if self.is_rank0 and cnt % log_iter == 0 and cnt != 0 and self.writer is not None:
                        self.writer.add_scalar('Avg_Batch_Loss_GradGrafting', avg_batch_loss_rrl / log_iter, cnt)
                        edge_p = self.edge_penalty().item()
                        self.writer.add_scalar('Edge_penalty/Log', np.log(edge_p), cnt)
                        self.writer.add_scalar('Edge_penalty/Origin', edge_p, cnt)
                        avg_batch_loss_rrl = 0.0

                optimizer.step()
                
                if self.is_rank0:
                    for param in self.net.parameters():
                        abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                        abs_gradient_avg += torch.sum(torch.abs(param.grad)) / param.grad.numel()
                self.clip()

                if self.is_rank0 and (cnt % (TEST_CNT_MOD * (1 if self.save_best else 10)) == 0):
                    if valid_loader is not None:
                        acc_b, f1_b = self.test(test_loader=valid_loader, set_name='Validation')
                    else:
                        acc_b, f1_b = self.test(test_loader=data_loader, set_name='Training')
                    
                    if self.save_best and (f1_b > self.best_f1 or (np.abs(f1_b - self.best_f1) < 1e-10 and self.best_loss > epoch_loss_rrl)):
                        self.best_f1 = f1_b
                        self.best_loss = epoch_loss_rrl
                        self.save_model()
                    
                    accuracy_b.append(acc_b)
                    f1_score_b.append(f1_b)
                    if self.writer is not None:
                        self.writer.add_scalar('Accuracy_RRL', acc_b, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_RRL', f1_b, cnt // TEST_CNT_MOD)
            if self.is_rank0:
                logging.info('epoch: {}, loss_rrl: {}'.format(epo, epoch_loss_rrl))
                if self.writer is not None:
                    self.writer.add_scalar('Training_Loss_RRL', epoch_loss_rrl, epo)
                    self.writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                    self.writer.add_scalar('Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo)
        if self.is_rank0 and not self.save_best:
            self.save_model()
        return epoch_histc

    @torch.no_grad()
    def test(self, test_loader=None, set_name='Validation'):
        if test_loader is None:
            raise Exception("Data loader is unavailable!")
        
        y_list = []
        for X, y in test_loader:
            y_list.append(y)
        y_true = torch.cat(y_list, dim=0)
        y_true = y_true.cpu().numpy().astype(np.int64)
        y_true = np.argmax(y_true, axis=1)
        data_num = y_true.shape[0]

        slice_step = data_num // 40 if data_num >= 40 else 1
        logging.debug('y_true: {} {}'.format(y_true.shape, y_true[:: slice_step]))

        y_pred_b_list = []
        for X, y in test_loader:
            X = X.cuda(self.device_id, non_blocking=True)
            output = self.net.forward(X)
            y_pred_b_list.append(output)

        y_pred_b = torch.cat(y_pred_b_list).cpu().numpy()
        y_pred_b_arg = np.argmax(y_pred_b, axis=1)
        logging.debug('y_rrl_: {} {}'.format(y_pred_b_arg.shape, y_pred_b_arg[:: slice_step]))
        logging.debug('y_rrl: {} {}'.format(y_pred_b.shape, y_pred_b[:: (slice_step)]))

        accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg)
        f1_score_b = metrics.f1_score(y_true, y_pred_b_arg, average='macro')

        logging.info('-' * 60)
        logging.info('On {} Set:\n\tAccuracy of RRL  Model: {}'
                        '\n\tF1 Score of RRL  Model: {}'.format(set_name, accuracy_b, f1_score_b))
        logging.info('On {} Set:\nPerformance of  RRL Model: \n{}\n{}'.format(
            set_name, metrics.confusion_matrix(y_true, y_pred_b_arg), metrics.classification_report(y_true, y_pred_b_arg)))
        logging.info('-' * 60)

        return accuracy_b, f1_score_b

    def save_model(self):
        rrl_args = {'dim_list': self.dim_list, 'use_not': self.use_not, 'use_skip': self.use_skip, 'estimated_grad': self.estimated_grad, 
                    'use_nlaf': self.use_nlaf, 'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}
        torch.save({'model_state_dict': self.net.state_dict(), 'rrl_args': rrl_args}, self.save_path)

    def detect_dead_node(self, data_loader=None):
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_size, dtype=torch.double, device=self.device_id)
                layer.forward_tot = 0

            for x, y in data_loader:
                x_bar = x.cuda(self.device_id)
                self.net.bi_forward(x_bar, count=True)

    def rule_print(self, feature_name, label_name, train_loader, file=sys.stdout, mean=None, std=None, display=True):
        if self.net.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")

        if self.net.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        self.net.layer_list[0].get_bound_name(feature_name, mean, std)

        for i in range(1, len(self.net.layer_list) - 1):
            layer = self.net.layer_list[i]
            layer.get_rules(layer.conn.prev_layer, layer.conn.skip_from_layer)
            skip_rule_name = None if layer.conn.skip_from_layer is None else layer.conn.skip_from_layer.rule_name
            wrap_prev_rule = i != 1
            layer.get_rule_description((skip_rule_name, layer.conn.prev_layer.rule_name), wrap=wrap_prev_rule)

        layer = self.net.layer_list[-1]
        layer.get_rule2weights(layer.conn.prev_layer, layer.conn.skip_from_layer)
        
        if not display:
            return layer.rule2weights
        
        print('RID', end='\t', file=file)
        for i, ln in enumerate(label_name):
            print('{}(b={:.4f})'.format(ln, layer.bl[i]), end='\t', file=file)
        print('Support\tRule', file=file)
        for rid, w in layer.rule2weights:
            print(rid, end='\t', file=file)
            for li in range(len(label_name)):
                print('{:.4f}'.format(w[li]), end='\t', file=file)
            now_layer = self.net.layer_list[-1 + rid[0]]
            print('{:.4f}'.format((now_layer.node_activation_cnt[layer.rid2dim[rid]] / now_layer.forward_tot).item()),
                  end='\t', file=file)
            print(now_layer.rule_name[rid[1]], end='\n', file=file)
        print('#' * 60, file=file)
        return layer.rule2weights