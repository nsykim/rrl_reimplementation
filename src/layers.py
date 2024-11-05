import torch
import torch.nn as nn
from collections import defaultdict

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10
INIT_L = 0.0

class Product(torch.autograd.Function):
    """Custom tensor product function class"""
    @staticmethod
    def forward(ctx, X):
        """ Performs the forward pass of the custom layer """
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """ Computes the gradient of the loss with respect to the input tensors """
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class EstimatedProduct(torch.autograd.Function):
    """ Custom tensor product function class with an estimated derivative """
    @staticmethod
    def forward(ctx, X):
        """ Performs the forward pass of the custom layer """
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """ Computes the gradient of the loss with respect to the input tensors """
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * ((-1. / (-1. + torch.log(y.unsqueeze(1) ** 2))) / (X + EPSILON))
        return grad_input



class GradientGraft(torch.autograd.Function):
    """ Custom gradient grafting class for the RRL """
    @staticmethod
    def forward(ctx, X, Y):
        """ Performs the forward pass of the layer """
        ctx.save_for_backward(X, Y)
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        """ Performs the backward pass for the custom layer """
        return None, grad_output.clone()

class Binarize(torch.autograd.Function):
    """ Custom binarization using sigmoid function """
    @staticmethod
    def forward(ctx, X):
        """ Applies a sigmoid activation function to the input tensor and thresholds the result """
        y = torch.sigmoid(X) > 0.5
        return y.to(torch.float)
    
    @staticmethod
    def backward(ctx, grad_output):
        """ Performs the backward pass for a custom layer """
        X, y = ctx.saved_tensors
        grad_input = grad_output *  y * (1-y)
        return grad_input

class FeatureBinarizer(nn.Module):
    """
    Custom layer that binarizes continuous features into multiple bins and combines them with discrete features.
    Args:
        - num_bins (int): The number of bins to use for binarizing continuous features.
        - input_shape (tuple): The shape of the input features.
        - min_val (torch.Tensor): The minimum value for bin centers. Default is None.
        - max_val (torch.Tensor): The maximum value for bin centers. Default is None.

    """
    def __init__(self, num_bins, input_shape, min_val=None, max_val=None):
        super(FeatureBinarizer, self).__init__()
        self.num_bins = num_bins
        self.input_shape = input_shape
        self.discrete_feature_count = input_shape[0]
        self.continuous_feature_count = input_shape[1]
        self.output_dim = self.discrete_feature_count + 2 * num_bins * self.continuous_feature_count
        self.feature_mapping = {i: i for i in range(self.output_dim)}
        self.layer_type = 'binarization'
        self.min_val = nn.Parameter(min_val, requires_grad=False) if min_val is not None else None
        self.max_val = nn.Parameter(max_val, requires_grad=False) if max_val is not None else None
        self.dimIDs = {i: i for i in range(self.output_dim)}

        if self.continuous_feature_count > 0:
            bin_centers = self._initialize_bin_centers()
            self.bin_centers = nn.Parameter(bin_centers, requires_grad=False)

    def _initialize_bin_centers(self):
        """
        Initializes the bin centers for continuous features.

        If `min_val` and `max_val` are specified, the bin centers are initialized
        uniformly within the range [`min_val`, `max_val`]. Otherwise, they are
        initialized using a standard normal distribution.
        """
        if self.min_val is not None and self.max_val is not None:
            return self.min_val + torch.rand(self.num_bins, self.continuous_feature_count) * (self.max_val - self.min_val)
        return torch.randn(self.num_bins, self.continuous_feature_count)

    def forward(self, input_data):
        """ Forward pass for the custom layer """
        if self.continuous_feature_count > 0:
            discrete_part, continuous_part = input_data[:, :self.input_shape[0]], input_data[:, self.input_shape[0]:]
            continuous_part = continuous_part.unsqueeze(-1)
            bin_diff = continuous_part - self.bin_centers.t()
            bin_results = Binarize.apply(bin_diff).view(continuous_part.shape[0], -1)
            combined_feats = torch.cat((discrete_part, bin_results, 1. - bin_results), dim=1)
            return combined_feats
        
        return input_data

    @torch.no_grad()
    def binarized_forward(self, input_data):
        """ Perform a binarized forward pass on the input data """
        return self.forward(input_data)

    def clip_weights(self):
        """ Clips the weights of the bin centers to be within the specified minimum and maximum values """
        if self.continuous_feature_count > 0 and self.min_val is not None and self.max_val is not None:
            self.bin_centers.data = torch.clamp(self.bin_centers.data, self.min_val, self.max_val)

    def generate_feature_names(self, feature_names, mean=None, std=None):
        """ Generates feature names with bin centers and operators """
        feature_labels = []
        for i in range(self.discrete_feature_count):
            feature_labels.append(feature_names[i])
        if self.continuous_feature_count > 0:
            for c, op in zip(self.bin_centers.t(), ('<', '>')):
                c = c.detach().cpu().numpy()
                for i, center in enumerate(c.T):
                    fi_name = feature_names[self.discrete_feature_count + i]
                    for j in center:
                        if mean is not None and std is not None:
                            j = j * std[i] + mean[i]
                        feature_labels.append('{} {} {:.3f}'.format(fi_name, op, j))
        self.rule_name = feature_labels
        return self.rule_name

class LinearRegressionLayer(nn.Module):
    """
    Custom linear regression layer
    Args:
        - num_outputs (int): The number of output features.
        - input_shape (int): The shape of the input features.
    """
    def __init__(self, num_outputs, input_shape):
        """ Initializes the LinearRegressionLayer """
        super(LinearRegressionLayer, self).__init__()
        self.num_outputs = num_outputs
        self.input_shape = input_shape
        self.output_dim = num_outputs
        self.rid2dim = None # rule id to dimension id
        self.rule2weights = None # rule to weights
        self.layer_type = 'linear'
        
        self.linear = nn.Linear(self.input_shape, self.output_dim)

    def forward(self, inputs):
        """ Perform a forward pass through the layer """
        x = self.linear(inputs)
        return x

    @torch.no_grad()
    def binarized_forward(self, inputs):
        """ Perform a binarized forward pass through the layer """
        return self.forward(inputs)

    def clip(self):
        """ Clip the weights of the linear layer """
        for param in self.linear.parameters():
            param.data.clamp_(-1.0, 1.0)

    def compute_l1_norm(self):
        """ Compute the L1 norm of the weights """
        return torch.norm(self.linear.weight, p=1)
    
    def compute_l2_norm(self):
        """ Compute the L2 norm of the weights """ 
        return torch.sum(self.linear.weight ** 2)
    
    def calculate_rule_weights(self, prev_layer, skip_connect_layer):
        """ Calculate the weights of the rules """
        prev_layer = self.conn.prev_layer
        skip_connect_layer = self.conn.skip_from_layer

        # mark skipped layers, then shift IDs and weights to accomodate the skips
        always_act_pos = (prev_layer.activation_nodes == prev_layer.forward_tot) # always active position
        merged_dimIDs = prev_dimIDs = {k: (-1, v) for k, v in prev_layer.dimIDs.items()}
        if skip_connect_layer is not None:
            shifted_dimIDs = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dimIDs.items()}
            merged_dimIDs = defaultdict(lambda: -1, {**shifted_dimIDs, **prev_dimIDs})
            always_act_pos = torch.cat(
                [always_act_pos, (skip_connect_layer.activation_nodes == skip_connect_layer.forward_tot)])
        
        # get linear weights and biases
        Wl, bl = list(self.linear.parameters()) 
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl # add bias for always active nodes
        Wl = Wl.cpu().detach().numpy()
        self.bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float)) # dict for rule weights
        rid2dim = {} # dict for rule ids to dims
        for label_id, wl in enumerate(Wl): # for each label's weights
            for i, w in enumerate(wl):
                rid = merged_dimIDs[i]
                if rid == -1 or rid[1] == -1: # if the rule id is invalid, skip
                    continue
                marked[rid][label_id] += w # add weight to the rule
                rid2dim[rid] = i % prev_layer.output_dim # map rule id to dimension id

        self.rid2dim = rid2dim # save rule id to dimension id mapping
        # sort rules by the maximum absolute value of their weights
        self.rule2weights = sorted(marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True)


class ConjunctionLayer(nn.Module):
    """
    A custom neural network layer that performs conjunction operations on the input.
    Args:
        num_conjunctions (int): The number of conjunctions to be performed.
        input_shape (int): The shape of the input tensor.
        alpha (float, optional): A parameter for the novel conjunction operation. Default is 0.999.
        beta (float, optional): A parameter for the novel conjunction operation. Default is 8.
        gamma (float, optional): A parameter for the novel conjunction operation. Default is 1.
    """

    def __init__(self, num_conjunctions, input_shape, alpha=0.999, beta=8, gamma=1):
        """ Initializes the ConjunctionLayer """
        super(ConjunctionLayer, self).__init__()
        self.num_conjunctions = num_conjunctions
        self.input_shape = input_shape
        self.output_dim = num_conjunctions
        self.layer_type = 'conjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_shape, self.output_dim))

        self.activation_cnt = None

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs):
        """ Perform a forward pass through the layer using novel activation functions """
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        """ Perform a continuous forward pass through the layer using novel activation functions """
        inputs = 1.- inputs
        x1 = (1. - 1. / (1. - (inputs * self.alpha) ** self.beta))
        w1 = (1. - 1. / (1. - (self.weights * self.alpha) ** self.beta))
        return 1. / (1. + x1 @ w1) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, inputs):
        """ Perform a binarized forward pass through the layer using novel activation functions """
        binary_weights = Binarize.apply(self.weights - THRESHOLD)
        res = (1 - inputs) @ binary_weights
        return torch.where(res > 0, torch.zeros_like(res), torch.ones_like(res))  

    def clip_weights(self):
        """ Clip the weights of the layer """
        self.weights.data.clamp_(INIT_L, 1.0)

class DisjunctionLayer(nn.Module):
    """
    A custom neural network layer that performs disjunction operations on the input.
    Args:
        num_disjunctions (int): The number of disjunctions to be performed.
        input_shape (int): The shape of the input tensor.
        alpha (float, optional): A parameter for the novel disjunction operation. Default is 0.999.
        beta (float, optional): A parameter for the novel disjunction operation. Default is 8.
        gamma (float, optional): A parameter for the novel disjunction operation. Default is 1.
    """
    def __init__(self, num_disjunctions, input_shape, alpha=0.999, beta=8, gamma=1):
        """ Initializes the DisjunctionLayer """
        super(DisjunctionLayer, self).__init__()
        self.num_disjunctions = num_disjunctions
        self.input_shape = input_shape 
        self.output_dim = num_disjunctions
        self.layer_type = 'disjunction'

        self.weights = nn.Parameter(INIT_L + (0.5 - INIT_L) * torch.rand(self.input_shape, self.num_disjunctions))

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs):
        """ Perform a forward pass through the layer using novel activation functions """
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        """ Perform a continuous forward pass through the layer using novel activation functions """
        x1 = (1. - 1. / (1. - (inputs * self.alpha) ** self.beta))
        w1 = (1. - 1. / (1. - (self.weights * self.alpha) ** self.beta))
        return 1. / (1. + x1 @ w1) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, inputs):
        """ Perform a binarized forward pass through the layer using novel activation functions """
        binary_weights = Binarize.apply(self.weights - THRESHOLD)
        res = inputs @ binary_weights
        return torch.where(res > 0, torch.ones_like(res), torch.zeros_like(res))

    def clip_weights(self):
        """ Clip the weights of the layer """
        self.weights.data.clamp_(INIT_L, 1.0)

class OriginalConjunctionLayer(nn.Module):
    """
    A PyTorch layer that performs a standard conjunction operation on the input tensor.
    Args:
        num_conjunctions (int): The number of conjunctions to be learned.
        input_shape (int): The shape of the input tensor.
        stochastic_grad (bool, optional): If True, use EstimatedProduct for stochastic gradient estimation. 
                                          If False, use Product. Default is False.
    """

    def __init__(self, num_conjunctions, input_shape, stochastic_grad=False):
        """ Initializes the OriginalConjunctionLayer """
        super(OriginalConjunctionLayer, self).__init__()
        self.num_conjunctions = num_conjunctions
        self.input_shape = input_shape
        self.output_dim = self.num_conjunctions
        self.layer_type = 'conjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_shape, self.num_conjunctions))
        self.prod = EstimatedProduct if stochastic_grad else Product

    def forward(self, inputs):
        """ Perform a forward pass through the layer """
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        """ Perform a continuous forward pass through the layer """
        return self.prod(1 - (1 - inputs).unsqueeze(-1) * self.weights)

    @torch.no_grad()
    def binarized_forward(self, inputs):
        """ Perform a binarized forward pass through the layer """
        binarized_weights = Binarize.apply(self.weights - THRESHOLD)
        return torch.prod(1 - (1 - inputs).unsqueeze(-1) * binarized_weights, dim=1)

    def clip_weights (self):
        """ Clip the weights of the layer """
        self.weights.data.clamp_(0.0, 1.0)

class OriginalDisjunctionLayer(nn.Module):
    """ 
    A PyTorch layer that performs a standard disjunction operation on the input tensor 
    Args:
        num_disjunctions (int): The number of disjunctions to be learned.
        input_shape (int): The shape of the input tensor.
        stochastic_grad (bool, optional): If True, use EstimatedProduct for stochastic gradient estimation.
    """
    def __init__(self, num_disjunctions, input_shape, stochastic_grad=False):
        """ Initializes the OriginalDisjunctionLayer """
        super(OriginalDisjunctionLayer, self).__init__()
        self.num_disjunctions = num_disjunctions
        self.input_shape = input_shape
        self.output_dim = num_disjunctions
        self.layer_type = 'disjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_shape, num_disjunctions))
        self.prod = EstimatedProduct if stochastic_grad else Product

    def forward(self, inputs):
        """ Perform a forward pass through the layer """
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        """ Perform a continuous forward pass through the layer """
        return 1 - self.prod.apply(1 - inputs.unsqueeze(-1) * self.weights)

    @torch.no_grad()
    def binarized_forward(self, inputs):
        """ Perform a binarized forward pass through the layer """
        binarized_weights = Binarize.apply(self.weights - THRESHOLD)
        return 1 - torch.prod(1 - inputs.unsqueeze(-1) * binarized_weights, dim=1)

    def clip_weights(self):
        """" Clip the weights of the layer """
        self.weights.data.clamp_(0.0, 1.0)


def extract_rules(previous_layer, skip_connection_layer, current_layer, position_shift=0):
    """
    Extracts rules from the current layer based on the weights and activation nodes, 
    considering connections from the previous layer and an optional skip connection layer.
    """

    dimIDs = defaultdict(lambda: -1)
    rules = {}
    rule_counter = 0
    rule_list = []

    # binarized_weights shape = (number_of_nodes, input_shape)
    binarized_weights = (current_layer.weights.t() > 0.5).type(torch.int).detach().cpu().numpy()

    # Merge dimIDs from the previous layer and skip connection layer (if available)
    merged_node_map = previous_node_map = {key: (-1, val) for key, val in previous_layer.dimIDs.items()}
    if skip_connection_layer is not None: # if we have skips, make sure to shift everything as needed
        shifted_node_map = {
            (key + previous_layer.output_dim): (-2, val) 
            for key, val in skip_connection_layer.dimIDs.items()
        }
        merged_node_map = defaultdict(lambda: -1, {**shifted_node_map, **previous_node_map})

    for node_index, weights_row in enumerate(binarized_weights):
        # Skip nodes that are inactive (dead nodes) or fully active (always triggered)
        if current_layer.activation_nodes[node_index + position_shift] == 0 or \
           current_layer.activation_nodes[node_index + position_shift] == current_layer.forward_tot:
            dimIDs[node_index + position_shift] = -1
            continue
        
        rule = {}
        # MAPPED WITH FOLLOWING CONVENTION rule[i] = {k, rule_id}
        # k == -1 connects to rule in previous layer
        # k == 1 connects to a rule in previous layer (NOT)
        # k == -2 connects to a rule in skip connection layer
        # k == 2 connects to a rule in skip connection layer (NOT)
        feature_bounds = {}

        # Special handling for binarization layers to account for discrete features
        if previous_layer.layer_type == 'binarization' and previous_layer.input_shape[1] > 0:
            discrete_features = torch.cat((previous_layer.cl.t().reshape(-1), previous_layer.cl.t().reshape(-1))).detach().cpu().numpy()

        for weight_index, weight in enumerate(weights_row):
            negate_input = 1
            # Only consider positive weights and valid previous mappings
            if weight > 0 and merged_node_map[weight_index][1] != -1:
                if previous_layer.layer_type == 'binarization' and weight_index >= previous_layer.discrete_feature_count:
                    # Handle discrete input features
                    feature_index = weight_index - previous_layer.discrete_feature_count
                    bin_index = feature_index // previous_layer.n
                    if bin_index not in feature_bounds:
                        feature_bounds[bin_index] = [weight_index, discrete_features[feature_index]]
                        rule[(-1, weight_index)] = 1  # Input connected to the previous layer's node
                    else:
                        # Merge bounds for a given feature
                        if (feature_index < discrete_features.shape[0] // 2 and current_layer.layer_type == 'conjunction') or \
                           (feature_index >= discrete_features.shape[0] // 2 and current_layer.layer_type == 'disjunction'):
                            update_function = max
                        else:
                            update_function = min
                        feature_bounds[bin_index][1] = update_function(feature_bounds[bin_index][1], discrete_features[feature_index])
                        if feature_bounds[bin_index][1] == discrete_features[feature_index]:
                            del rule[(-1, feature_bounds[bin_index][0])]
                            rule[(-1, weight_index)] = 1
                            feature_bounds[bin_index][0] = weight_index
                else:
                    # Connect to the rule or node in the previous or skip connection layer
                    previous_rule_id = merged_node_map[weight_index]
                    rule[(previous_rule_id[0] * negate_input, previous_rule_id[1])] = 1
        
        # Assign unique rule IDs and store in dimIDs
        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = rule_counter
            rule_list.append(rule)
            dimIDs[node_index + position_shift] = rule_counter
            rule_counter += 1
        else:
            dimIDs[node_index + position_shift] = rules[rule]

    return dimIDs, rule_list

class UnionLayer(nn.Module):
    """
    A neural network layer that combines conjunction and disjunction operations.
    Args:
        units (int): Number of conjunctions and disjunctions.
        input_shape (tuple): Shape of the input tensor.
        use_novel_activation (bool, optional): Whether to use novel activation functions. Defaults to False.
        estimated_grad (bool, optional): Whether to use estimated gradients. Defaults to False.
        alpha (float, optional): Hyperparameter for novel activation. Defaults to 0.999.
        beta (int, optional): Hyperparameter for novel activation. Defaults to 8.
        gamma (int, optional): Hyperparameter for novel activation. Defaults to 1.
    """

    def __init__(self, units, input_shape, use_novel_activation=False, estimated_grad=False, alpha=0.999, beta=8, gamma=1):
        """ Initializes the UnionLayer """
        super(UnionLayer, self).__init__()
        self.units = units
        self.input_shape = input_shape
        self.output_dim = self.units * 2  # Union of conjunction and disjunction
        self.layer_type = 'union'
        self.activation_nodes = None
        self.rules = None
        self.rule_name = None
        self.dimIDs = None
        
        if use_novel_activation: 
            self.conjunction_layer = ConjunctionLayer(num_conjunctions=units, input_shape=input_shape, alpha=alpha, beta=beta, gamma=gamma)
            self.disjunction_layer = DisjunctionLayer(num_disjunctions=units, input_shape=input_shape, alpha=alpha, beta=beta, gamma=gamma)
        else:
            self.conjunction_layer = OriginalConjunctionLayer(num_conjunctions=units, input_shape=input_shape, stochastic_grad=estimated_grad)
            self.disjunction_layer = OriginalDisjunctionLayer(num_disjunctions=units, input_shape=input_shape, stochastic_grad=estimated_grad)

    def forward(self, input_tensor):
        """ Perform a forward pass through the layer """
        return torch.cat([self.conjunction_layer(input_tensor), self.disjunction_layer(input_tensor)], dim=1)

    def binarized_forward(self, input_tensor):
        """ Perform a binarized forward pass through the layer """
        return torch.cat([self.conjunction_layer.binarized_forward(input_tensor), 
                          self.disjunction_layer.binarized_forward(input_tensor)], dim=1)

    def edge_count(self):
        """ Compute the number of edges in the layer """
        return self._sum_binarized_weights(self.conjunction_layer) + self._sum_binarized_weights(self.disjunction_layer)

    def _sum_binarized_weights(self, layer):
        """ Compute the sum of binarized weights for a given layer """
        binarized_weights = Binarize.apply(layer.weights - THRESHOLD)
        return torch.sum(binarized_weights)

    def compute_l1_norm(self):
        """ Compute the L1 norm of the weights """
        return torch.sum(self.conjunction_layer.weights) + torch.sum(self.disjunction_layer.weights)

    def compute_l2_norm(self):
        """ Compute the L2 norm of the weights """
        return torch.sum(self.conjunction_layer.weights ** 2) + torch.sum(self.disjunction_layer.weights ** 2)

    def clip_weights(self):
        """ Clip the weights of the layer """
        self.conjunction_layer.clip_weights()
        self.disjunction_layer.clip_weights()

    def get_rules(self, previous_layer, skip_connection_layer):
        """ Extract rules from the conjunction and disjunction layers """
        self._sync_layer_stats()

        conjunction_dimIDs, conjunction_rules = extract_rules(previous_layer, skip_connection_layer, self.conjunction_layer)
        disjunction_dimIDs, disjunction_rules = extract_rules(previous_layer, skip_connection_layer, 
                                                                self.disjunction_layer, self.conjunction_layer.weights.shape[1])

        # how much to shift the id of the rules in the disjunction layer to put into one
        shift = max(conjunction_dimIDs.values()) + 1
        disjunction_dimIDs = {k: (-1 if v == -1 else v + shift) for k, v in disjunction_dimIDs.items()}

        # combine the rules and dimensionIDs
        self.dimIDs = defaultdict(lambda: -1, {**conjunction_dimIDs, **disjunction_dimIDs})
        self.rule_list = (conjunction_rules, disjunction_rules)

    def _sync_layer_stats(self):
        """ Synchronize the stats of the conjunction and disjunction layers """
        self.conjunction_layer.forward_tot = self.disjunction_layer.forward_tot = self.forward_tot
        self.conjunction_layer.activation_nodes = self.disjunction_layer.activation_nodes = self.activation_nodes

    def get_rule_description(self, input_rule_name, wrap=False):
        """ Generate rule descriptions for the layer """
        self.rule_name = []
        for rules, operator in zip(self.rule_list, ('&', '|')):
            self._append_rule_descriptions(rules, input_rule_name, operator, wrap)

    def _append_rule_descriptions(self, rules, input_rule_name, operator, wrap):
        """ Append rule descriptions to the rule_name list """
        for rule in rules:
            description = self._build_rule_description(rule, input_rule_name, operator, wrap)
            self.rule_name.append(description)

    def _build_rule_description(self, rule, input_rule_name, operator, wrap):
        """ Build a rule description from the rule """
        rule_description = ''
        for i, (layer_shift, rule_id) in enumerate(rule):
            prefix = f' {operator} ' if i != 0 else ''
            not_str = '~' if layer_shift > 0 else ''
            layer_shift *= -1 if layer_shift > 0 else 1
            var_name = input_rule_name[2 + layer_shift][rule_id]
            var_str = f'({var_name})' if wrap or not_str else var_name
            rule_description += f"{prefix}{not_str}{var_str}"
        return rule_description
