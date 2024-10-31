import torch
import torch.nn as nn
from collections import defaultdict
from sklearn import metrics
from torch.optim import lr_scheduler
import logging
import sys
import numpy as np

from src.layers import *

TEST_CNT_MOD = 500

class Net(nn.Module):
    """
    A neural network model with customizable layers and skip connections.
    Args:
        dim_list (list): List of dimensions for each layer.
        left (float, optional): Minimum value for feature binarization. Default is None.
        right (float, optional): Maximum value for feature binarization. Default is None.
        use_nlaf (bool, optional): Whether to use novel activation functions. Default is False.
        estimated_grad (bool, optional): Whether to use estimated gradients. Default is False.
        use_skip (bool, optional): Whether to use skip connections. Default is True.
        alpha (float, optional): Alpha parameter for UnionLayer. Default is 0.999.
        beta (float, optional): Beta parameter for UnionLayer. Default is 8.
        gamma (float, optional): Gamma parameter for UnionLayer. Default is 1.
        temperature (float, optional): Temperature parameter for the model. Default is 0.01.
    """
    def __init__(self, dim_list, left=None, right=None, use_nlaf=False, estimated_grad=False, use_skip=True, alpha=0.999, beta=8, gamma=1, temperature=0.01):
        """ Initializes the Net class """
        super(Net, self).__init__()
        self.dim_list = dim_list
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])
        self.use_skip = use_skip
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

        self._initialize_layers(dim_list, use_nlaf, estimated_grad, alpha, beta, gamma)

    def _initialize_layers(self, dim_list, use_nlaf, estimated_grad, alpha, beta, gamma):
        """ Initializes the layers of the model based on the provided dimensions and parameters """
        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)): # iterate over the dimensions
            num = prev_layer_dim 
            skip_from_layer = None
            if self.use_skip and i >= 4: # skip connections start from the 4th layer
                skip_from_layer = self.layer_list[-2]
                num += skip_from_layer.output_dim

            if i == 1: # first layer is the binerization layer
                layer = FeatureBinarizer(dim_list[i], num, min_val=self.left, max_val=self.right)
                layer_name = f'binary{i}'
            elif i == len(dim_list) - 1: # last layer is the linear regression layer
                layer = LinearRegressionLayer(dim_list[i], num)
                layer_name = f'lr{i}'
            else: # all other layers are union layers
                layer = UnionLayer(dim_list[i], num, use_novel_activation=use_nlaf, estimated_grad=estimated_grad, alpha=alpha, beta=beta, gamma=gamma)
                layer_name = f'union{i}'

            self._set_layer_connections(layer, skip_from_layer) # set the connections for the layers
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer) # add the layer to the model
            self.layer_list.append(layer)

    def _set_layer_connections(self, layer, skip_from_layer):
        """ Set the connections for a given layer """
        layer.conn = lambda: None
        layer.conn.prev_layer = self.layer_list[-1] if len(self.layer_list) > 0 else None
        layer.conn.is_skip_to_layer = False 
        layer.conn.skip_from_layer = skip_from_layer 
        if skip_from_layer is not None:
            skip_from_layer.conn.is_skip_to_layer = True

    def forward(self, x):
        """
        Forward pass through the network.

        The method iterates through each layer in `self.layer_list` and performs the following:
        - If the layer has a skip connection from another layer, concatenate the input tensor `x` with the tensor from the skip connection.
        - Call the forward method of the current layer.
        - If the layer has a skip connection to another layer, store the output tensor `x` for future use.

        Note:
            The skip connection tensors are deleted after they are used to free up memory.
        """
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None: 
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1) 
                del layer.conn.skip_from_layer.x_res
            x = layer(x) 
            if layer.conn.is_skip_to_layer: 
                layer.x_res = x 
        return x
    
    def binarized_forward(self, x, count=False):
        """
        Perform a forward pass through the network with binarized layers.

        Notes:
            - If a layer has a skip connection from another layer, concatenate the input tensor with the skip connection tensor.
            - If a layer has a skip connection to another layer, store the output tensor for the skip connection.
            - If `count` is True and the layer is not a linear layer, update the activation nodes and forward pass count.
        """
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None: 
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1) 
                del layer.conn.skip_from_layer.x_res
            x = layer.binarized_forward(x) 
            if layer.conn.is_skip_to_layer: 
                layer.x_res = x 
            if count and layer.layer_type != 'linear': 
                layer.activation_nodes += torch.sum(x, dim=0) 
                layer.forward_tot += x.shape[0] 
        return x


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """
    A custom implementation of `torch.nn.parallel.DistributedDataParallel` that exposes additional properties.
    Properties:
        layer_list (list): A list of layers from the wrapped module.
        t: A custom property from the wrapped module.
    """
    @property
    def layer_list(self):
        """ Returns the list of layers in the module """
        return self.module.layer_list
    
    @property
    def t(self):
        """ Returns the value of the attribute 't' from the 'module' object """
        return self.module.t


class RRL:
    """
    RRL (Rule-based Representational Learner) class for implementing a neural network with various features such as skip connections, 
    novel activation functions, and distributed data parallelism.
    Args:
        use_nlaf (bool): Flag to use novel activation functions.
        best_f1 (float): Best F1 score achieved during training.
        best_loss (float): Best loss achieved during training.
        device_id (int): ID of the GPU device to use.
        save_best (bool): Flag to save the best model during training.
        save_path (str): Path to save the model.
        writer (SummaryWriter): TensorBoard writer for logging.
        net (Net): The neural network model.
    """
    def __init__(self, dim_list, device_id, log_file=None, writer=None, left=None,
                 right=None, save_best=False, estimated_grad=False, save_path=None, distributed=True, use_skip=False, 
                 use_nlaf=False, alpha=0.999, beta=8, gamma=1, temperature=0.01):
        self.dim_list = dim_list
        self.use_skip = use_skip
        self.use_nlaf = use_nlaf # use novel activation functions
        self.alpha = alpha # alpha parameter for RRL 
        self.beta = beta # beta parameter for RRL
        self.gamma = gamma # gamma parameter for RRL
        self.best_f1 = -1.
        self.best_loss = 1e20

        self.device_id = device_id
        self.save_best = save_best
        self.estimated_grad = estimated_grad
        self.save_path = save_path
        self.writer = writer

        self._setup_logging(log_file)
        self.net = self._initialize_net(dim_list, left, right, use_nlaf, estimated_grad, use_skip, alpha, beta, gamma, temperature, distributed)

    def _setup_logging(self, log_file):
        """
        This method configures the logging settings by removing any existing handlers
        from the root logger and then adding a new handler based on the provided log_file parameter.
        If log_file is None, logs will be output to the standard output (stdout). Otherwise, logs
        will be written to the specified log file.
        """
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
        if log_file is None:
            logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)

    def _initialize_net(self, dim_list, left, right, use_nlaf, estimated_grad, use_skip, alpha, beta, gamma, temperature, distributed):
        """Initializes the neural network with the given parameters """
        net = Net(dim_list, left=left, right=right, use_nlaf=use_nlaf, estimated_grad=estimated_grad, use_skip=use_skip, alpha=alpha, beta=beta, gamma=gamma, temperature=temperature)
        net.cuda(self.device_id)
        if distributed:
            net = DistributedDataParallel(net, device_ids=[self.device_id])
        return net

    def clip(self):
        """ Clips the weights of each layer in the network except the last one """
        for layer in self.net.layer_list[:-1]:
            layer.clip_weights()
    
    def edge_penalty(self):
        """ Calculate the edge penalty for the network """
        return sum(layer.edge_count() for layer in self.net.layer_list[1:-1])

    
    def l1_penalty(self):
        """ Calculate the L1 penalty for the network layers """
        return sum(layer.compute_l1_norm() for layer in self.net.layer_list[1:])
    
    def l2_penalty(self):
        """ Computes the L2 penalty for the layers in the network to prevent overfitting by penalizing large weights """
        return sum(layer.compute_l2_norm() for layer in self.net.layer_list[1:])
    
    def mixed_penalty(self):
        """ Calculate the mixed penalty for the network layers """
        penalty = sum(layer.compute_l2_norm() for layer in self.net.layer_list[1:-1])
        penalty += self.net.layer_list[-1].compute_l1_norm()
        return penalty

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """ Adjusts the learning rate of the optimizer according to an exponential decay schedule """
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, data_loader=None, valid_loader=None, epoch=50, lr=0.01, lr_decay_epoch=100,lr_decay_rate=0.75, weight_decay=0.0, log_iter=50): 
        """ Train the model using the provided data loader and validation loader """
        if data_loader is None:
            raise Exception("Data loader is unavailable!")

        accuracy_b = []
        f1_score_b = []

        criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=0.0)

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
                X = X.cuda(self.device_id, non_blocking=True) # move input to GPU
                y = y.cuda(self.device_id, non_blocking=True) # move target to GPU
                optimizer.zero_grad() # zero the gradients
                
                y_bar = self.net.forward(X) / torch.exp(self.net.t) 
                y_arg = torch.argmax(y, dim=1) 
                
                loss_rrl = criterion(y_bar, y_arg) + weight_decay * self.l2_penalty() # compute loss with L2 penalty
                
                ba_loss_rrl = loss_rrl.item() # get loss value
                epoch_loss_rrl += ba_loss_rrl # accumulate loss for the epoch
                avg_batch_loss_rrl += ba_loss_rrl # accumulate loss for the batch
                 
                loss_rrl.backward()

                cnt += 1
                with torch.no_grad(): # update the weights
                    if cnt % log_iter == 0 and cnt != 0 and self.writer is not None:
                        self.writer.add_scalar('Avg_Batch_Loss_GradGrafting', avg_batch_loss_rrl / log_iter, cnt)
                        edge_p = self.edge_penalty().item()
                        self.writer.add_scalar('Edge_penalty/Log', np.log(edge_p), cnt)
                        self.writer.add_scalar('Edge_penalty/Origin', edge_p, cnt)
                        avg_batch_loss_rrl = 0.0

                optimizer.step()
                
                for param in self.net.parameters(): # clip the gradients
                    abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                    abs_gradient_avg += torch.sum(torch.abs(param.grad)) / param.grad.numel()
                self.clip()

                if cnt % (TEST_CNT_MOD * (1 if self.save_best else 10)) == 0:  # test the model
                    if valid_loader is not None:
                        logging.debug('Testing on Validation Set')
                        acc_b, f1_b = self.test(test_loader=valid_loader, set_name='Validation') 
                    else:
                        logging.debug('Testing on Training Set')
                        acc_b, f1_b = self.test(test_loader=data_loader, set_name='Training')
                    
                    if self.save_best and (f1_b > self.best_f1 or (np.abs(f1_b - self.best_f1) < 1e-10 and self.best_loss > epoch_loss_rrl)): # save the best model
                        self.best_f1 = f1_b
                        self.best_loss = epoch_loss_rrl
                        self.save_model()
                    
                    accuracy_b.append(acc_b)
                    f1_score_b.append(f1_b)
                    if self.writer is not None:
                        self.writer.add_scalar('Accuracy_RRL', acc_b, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_RRL', f1_b, cnt // TEST_CNT_MOD)
            logging.info('epoch: {}, loss_rrl: {}'.format(epo, epoch_loss_rrl))
            if self.writer is not None:
                self.writer.add_scalar('Training_Loss_RRL', epoch_loss_rrl, epo)
                self.writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                self.writer.add_scalar('Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo)
        if not self.save_best:
            self.save_model()
        return epoch_histc

    @torch.no_grad()
    def test(self, test_loader=None, set_name='Validation'):
        """ Tests the model using the provided test data loader and logs the performance metrics """
        if test_loader is None:
            raise Exception("Data loader is unavailable!")
        
        y_list = []
        for X, y in test_loader: # get the true labels
            y_list.append(y)
        y_true = torch.cat(y_list, dim=0)
        y_true = y_true.cpu().numpy().astype(np.int64)
        y_true = np.argmax(y_true, axis=1) 
        data_num = y_true.shape[0]

        slice_step = data_num // 40 if data_num >= 40 else 1
        logging.debug('y_true: {} {}'.format(y_true.shape, y_true[:: slice_step]))

        y_pred_b_list = []
        for X, y in test_loader:
            X = X.cuda(self.device_id, non_blocking=True) # move input to GPU
            output = self.net.forward(X)
            y_pred_b_list.append(output)

        y_pred_b = torch.cat(y_pred_b_list).cpu().numpy() # get the predicted labels
        y_pred_b_arg = np.argmax(y_pred_b, axis=1)
        logging.debug('y_rrl_: {} {}'.format(y_pred_b_arg.shape, y_pred_b_arg[:: slice_step]))
        logging.debug('y_rrl: {} {}'.format(y_pred_b.shape, y_pred_b[:: (slice_step)]))

        accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg) # calculate the accuracy
        f1_score_b = metrics.f1_score(y_true, y_pred_b_arg, average='macro')

        logging.info('-' * 60)
        logging.info('On {} Set:\n\tAccuracy of RRL  Model: {}'
                        '\n\tF1 Score of RRL  Model: {}'.format(set_name, accuracy_b, f1_score_b))
        logging.info('On {} Set:\nPerformance of  RRL Model: \n{}\n{}'.format(
            set_name, metrics.confusion_matrix(y_true, y_pred_b_arg), metrics.classification_report(y_true, y_pred_b_arg)))
        logging.info('-' * 60)

        return accuracy_b, f1_score_b

    def save_model(self):
        """
        Saves the current model state and relevant arguments to a file.

        This method prints a message indicating that the model is being saved,
        then saves the model's state dictionary and relevant arguments to the
        specified file path using PyTorch's `torch.save` function.

        The saved dictionary contains:
        - 'model_state_dict': The state dictionary of the model.
        - 'rrl_args': A dictionary of relevant arguments including:
            - 'dim_list': The list of dimensions.
            - 'use_skip': Whether to use skip connections.
            - 'estimated_grad': Whether to use estimated gradients.
            - 'use_nlaf': Whether to use non-linear activation functions.
            - 'alpha': The alpha parameter.
            - 'beta': The beta parameter.
            - 'gamma': The gamma parameter.
        """
        print('Saving model...')
        rrl_args = {'dim_list': self.dim_list, 'use_skip': self.use_skip, 'estimated_grad': self.estimated_grad, 
                    'use_nlaf': self.use_nlaf, 'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}
        torch.save({'model_state_dict': self.net.state_dict(), 'rrl_args': rrl_args}, self.save_path)

    def detect_dead_node(self, data_loader=None):
        """ Detects dead nodes in the neural network layers """
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.activation_nodes = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device_id)
                layer.forward_tot = 0

            for x, y in data_loader:
                x_bar = x.cuda(self.device_id)
                self.net.binarized_forward(x_bar, count=True)

    def rule_print(self, feature_name, label_name, train_loader, file=sys.stdout, mean=None, std=None, display=True):
        """ Prints or returns the rules and their corresponding weights for the neural network layers """
        if self.net.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")

        if self.net.layer_list[1].activation_nodes is None:
            self.detect_dead_node(train_loader)

        self.net.layer_list[0].generate_feature_names(feature_name, mean, std)

        for i in range(1, len(self.net.layer_list) - 1):
            layer = self.net.layer_list[i]
            layer.get_rules(layer.conn.prev_layer, layer.conn.skip_from_layer)
            skip_rule_name = None if layer.conn.skip_from_layer is None else layer.conn.skip_from_layer.rule_name
            wrap_prev_rule = i != 1
            layer.get_rule_description((skip_rule_name, layer.conn.prev_layer.rule_name), wrap=wrap_prev_rule)

        layer = self.net.layer_list[-1]
        layer.calculate_rule_weights(layer.conn.prev_layer, layer.conn.skip_from_layer)
        
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
            print('{:.4f}'.format((now_layer.activation_nodes[layer.rid2dim[rid]] / now_layer.forward_tot).item()),
                  end='\t', file=file)
            print(now_layer.rule_name[rid[1]], end='\n', file=file)
        print('#' * 60, file=file)
        return layer.rule2weights