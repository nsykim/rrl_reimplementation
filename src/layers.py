import torch
import torch.nn as nn
from collections import defaultdict

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10
INIT_L = 0.0

def augment_with_negation(x, use_negation):
    """Helper function to handle input augmentation with negation."""
    if use_negation:
        return torch.cat((x, 1 - x), dim=1)
    return x

class Product(torch.autograd.Function):
    """Tensor product function."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class EstimatedProduct(torch.autograd.Function):
    """Tensor product function with a estimated derivative."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * ((-1. / (-1. + torch.log(y.unsqueeze(1) ** 2))) / (X + EPSILON))
        return grad_input



class GradientGraft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class FeatureBinarizer(nn.Module):
    def __init__(self, num_bins, input_shape, use_negation=False, min_val=None, max_val=None):
        super(FeatureBinarizer, self).__init__()
        self.num_bins = num_bins
        self.input_shape = input_shape
        self.use_negation = use_negation
        self.discrete_feature_count = input_shape[0]
        self.continuous_feature_count = input_shape[1]
        self.output_dim = self.discrete_feature_count + 2 * num_bins * self.continuous_feature_count
        self.feature_mapping = {i: i for i in range(self.output_dim)}
        self.discrete_feature_count *= 2 if use_negation else 1
        self.layer_type = 'binarization'
        self.min_val = nn.Parameter(min_val, requires_grad=False) if min_val is not None else None
        self.max_val = nn.Parameter(max_val, requires_grad=False) if max_val is not None else None
        self.dimIDs = {i: i for i in range(self.output_dim)}

        if self.continuous_feature_count > 0:
            bin_centers = self._initialize_bin_centers()
            self.bin_centers = nn.Parameter(bin_centers, requires_grad=False)

    def _initialize_bin_centers(self):
        if self.min_val is not None and self.max_val is not None:
            return self.min_val + torch.rand(self.num_bins, self.continuous_feature_count) * (self.max_val - self.min_val)
        return torch.randn(self.num_bins, self.continuous_feature_count)

    def forward(self, input_data):
        discrete_part, continuous_part = input_data[:, :self.input_shape[0]], input_data[:, self.input_shape[0]:]

        discrete_part = augment_with_negation(discrete_part, self.use_negation)

        if self.continuous_feature_count > 0:
            continuous_part = continuous_part.unsqueeze(-1)
            bin_diff = continuous_part - self.bin_centers.t()
            binary_results = (bin_diff > 0).float().view(continuous_part.shape[0], -1)
            binary_neg_results = 1.0 - binary_results
            combined_features = torch.cat([discrete_part, binary_results, binary_neg_results], dim=1)
            return combined_features

        return discrete_part

    @torch.no_grad()
    def binarized_forward(self, input_data):
        return self.forward(input_data)

    def clip_weights(self):
        if self.continuous_feature_count > 0 and self.min_val is not None and self.max_val is not None:
            self.bin_centers.data = torch.clamp(self.bin_centers.data, self.min_val, self.max_val)

    def generate_feature_names(self, feature_names, mean=None, std=None):
        feature_labels = feature_names[:self.input_shape[0]]

        if self.use_negation:
            feature_labels += ['~' + name for name in feature_names[:self.input_shape[0]]]

        if self.continuous_feature_count > 0:
            for center, operator in [(self.bin_centers, '>'), (self.bin_centers, '<=')]:
                centers = center.detach().cpu().numpy()
                for i, bin_vals in enumerate(centers.T):
                    feature_name = feature_names[self.input_shape[0] + i]
                    for val in bin_vals:
                        if mean is not None and std is not None:
                            val = val * std[feature_name] + mean[feature_name]
                        feature_labels.append(f'{feature_name} {operator} {val:.3f}')

        self.rule_name = feature_labels
        return feature_labels

class LinearRegressionLayer(nn.Module):
    def __init__(self, num_outputs, input_shape):
        super(LinearRegressionLayer, self).__init__()
        self.num_outputs = num_outputs
        self.input_shape = input_shape
        self.output_dim = num_outputs
        self.rid2dim = None
        self.rule2weights = None
        self.layer_type = 'linear'
        
        self.linear = nn.Linear(self.input_shape, self.output_dim)

    def forward(self, inputs):
        x = self.linear(inputs)
        return x

    @torch.no_grad()
    def binarized_forward(self, inputs):
        return self.forward(inputs)

    def clip(self):
        for param in self.linear.parameters():
            param.data.clamp_(-1.0, 1.0)

    def compute_l1_norm(self):
        return torch.norm(self.linear.weight, p=1)
    
    def compute_l2_norm(self):
        return torch.sum(self.linear.weight ** 2)
    
    def calculate_rule_weights(self, prev_layer, skip_connect_layer):
        prev_layer = self.conn.prev_layer
        skip_connect_layer = self.conn.skip_from_layer

        always_act_pos = (prev_layer.activation_nodes == prev_layer.forward_tot)
        merged_dimIDs = prev_dimIDs = {k: (-1, v) for k, v in prev_layer.dimIDs.items()}
        if skip_connect_layer is not None:
            shifted_dimIDs = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dimIDs.items()}
            merged_dimIDs = defaultdict(lambda: -1, {**shifted_dimIDs, **prev_dimIDs})
            always_act_pos = torch.cat(
                [always_act_pos, (skip_connect_layer.activation_nodes == skip_connect_layer.forward_tot)])
        
        Wl, bl = list(self.linear.parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        self.bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dimIDs[i]
                if rid == -1 or rid[1] == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim

        self.rid2dim = rid2dim
        self.rule2weights = sorted(marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True)


class ConjunctionLayer(nn.Module):
    def __init__(self, num_conjunctions, input_shape, use_negation=False, alpha=0.999, beta=8, gamma=1, stochastic_grad=False):
        super(ConjunctionLayer, self).__init__()
        self.num_conjunctions = num_conjunctions
        self.use_negation = use_negation
        self.input_shape = input_shape * (2 if use_negation else 1)
        self.output_dim = num_conjunctions
        self.layer_type = 'conjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_shape, self.output_dim))

        self.activation_cnt = None

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs):
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)

        inputs = 1.- inputs
        x1 = (1. - 1. / (1. - (inputs * self.alpha) ** self.beta))
        w1 = (1. - 1. / (1. - (self.weights * self.alpha) ** self.beta))
        return 1. / (1. + x1 @ w1) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)

        binary_weights = Binarize.apply(self.weights - THRESHOLD)
        res = (1 - inputs) @ binary_weights
        return torch.where(res > 0, torch.zeros_like(res), torch.ones_like(res))  

    def clip_weights(self):
        self.weights.data.clamp_(INIT_L, 1.0)

class DisjunctionLayer(nn.Module):
    def __init__(self, num_disjunctions, input_shape, use_negation=False, alpha=0.999, beta=8, gamma=1, stochastic_grad=False):
        super(DisjunctionLayer, self).__init__()
        self.num_disjunctions = num_disjunctions
        self.use_negation = use_negation
        self.input_shape = input_shape * (2 if use_negation else 1)
        self.output_dim = num_disjunctions
        self.layer_type = 'disjunction'

        self.weights = nn.Parameter(INIT_L + (0.5 - INIT_L) * torch.rand(self.input_shape, self.num_disjunctions))

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs):
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)
        x1 = (1. - 1. / (1. - (inputs * self.alpha) ** self.beta))
        w1 = (1. - 1. / (1. - (self.weights * self.alpha) ** self.beta))
        return 1. / (1. + x1 @ w1) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)

        binary_weights = Binarize.apply(self.weights - THRESHOLD)
        res = inputs @ binary_weights
        return torch.where(res > 0, torch.ones_like(res), torch.zeros_like(res))

    def clip_weights(self):
        self.weights.data.clamp_(INIT_L, 1.0)

class OriginalConjunctionLayer(nn.Module):
    def __init__(self, n, input_shape, use_negation=False, stochastic_grad=False):
        super(OriginalConjunctionLayer, self).__init__()
        self.n = n
        self.use_negation = use_negation
        self.input_shape = input_shape * (2 if use_negation else 1)
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_shape, self.n))
        self.prod = EstimatedProduct if stochastic_grad else Product

    def forward(self, inputs):
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)
        return self.prod(1 - (1 - inputs).unsqueeze(-1) * self.weights)

    @torch.no_grad()
    def binarized_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)
        binarized_weights = Binarize.apply(self.weights - THRESHOLD)
        return torch.prod(1 - (1 - inputs).unsqueeze(-1) * binarized_weights, dim=1)

    def clip(self):
        self.weights.data.clamp_(0.0, 1.0)

class OriginalDisjunctionLayer(nn.Module):
    def __init__(self, n, input_shape, use_negation=False, stochastic_grad=False):
        super(OriginalDisjunctionLayer, self).__init__()
        self.n = n
        self.use_negation = use_negation
        self.input_shape = input_shape * (2 if use_negation else 1)
        self.output_dim = n
        self.layer_type = 'disjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_shape, n))
        self.prod = EstimatedProduct if stochastic_grad else Product

    def forward(self, inputs):
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)
        return 1 - self.prod(1 - inputs)

    @torch.no_grad()
    def binarized_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)
        binarized_weights = Binarize.apply(self.weights - THRESHOLD)
        return 1 - torch.prod(1 - inputs.unsqueeze(-1) * binarized_weights, dim=1)

    def clip(self):
        self.weights.data.clamp_(0.0, 1.0)


def extract_rules(previous_layer, skip_connection_layer, current_layer, position_shift=0):
    dimIDs = defaultdict(lambda: -1)
    rules = {}
    rule_counter = 0
    rule_list = []

    # binarized_weights shape = (number_of_nodes, input_shape)
    binarized_weights = (current_layer.weights.t() > 0.5).type(torch.int).detach().cpu().numpy()

    # Merge dimIDs from the previous layer and skip connection layer (if available)
    merged_node_map = previous_node_map = {key: (-1, val) for key, val in previous_layer.dimIDs.items()}
    if skip_connection_layer is not None:
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
            # Handling for 'NOT' logic in input dimensions
            negate_input = 1
            if current_layer.use_negation:
                if weight_index >= current_layer.input_shape // 2:
                    negate_input = -1
                weight_index = weight_index % (current_layer.input_shape // 2)

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
    def __init__(self, units, input_shape, use_negation=False, use_novel_activation=False, estimated_grad=False, alpha=0.999, beta=8, gamma=1):
        super(UnionLayer, self).__init__()
        self.units = units
        self.use_negation = use_negation
        self.input_shape = input_shape
        self.output_dim = self.units * 2  # Union of conjunction and disjunction
        self.layer_type = 'union'
        self.activation_nodes = None
        self.rules = None
        self.rule_name = None
        self.dimIDs = None
        
        if use_novel_activation:
            self.conjunction_layer = ConjunctionLayer(num_conjunctions=units, input_shape=input_shape, use_negation=use_negation, alpha=alpha, beta=beta, gamma=gamma)
            self.disjunction_layer = DisjunctionLayer(num_disjunctions=units, input_shape=input_shape, use_negation=use_negation, alpha=alpha, beta=beta, gamma=gamma)
        else:
            self.conjunction_layer = OriginalConjunctionLayer(n=units, input_shape=input_shape, use_negation=use_negation, stochastic_grad=estimated_grad)
            self.disjunction_layer = OriginalDisjunctionLayer(n=units, input_shape=input_shape, use_negation=use_negation, stochastic_grad=estimated_grad)

    def forward(self, input_tensor):
        return torch.cat([self.conjunction_layer(input_tensor), self.disjunction_layer(input_tensor)], dim=1)

    def binarized_forward(self, input_tensor):
        return torch.cat([self.conjunction_layer.binarized_forward(input_tensor), 
                          self.disjunction_layer.binarized_forward(input_tensor)], dim=1)

    def edge_count(self):
        return self._sum_binarized_weights(self.conjunction_layer) + self._sum_binarized_weights(self.disjunction_layer)

    def _sum_binarized_weights(self, layer):
        binarized_weights = Binarize.apply(layer.weights - THRESHOLD)
        return torch.sum(binarized_weights)

    def compute_l1_norm(self):
        return torch.sum(self.conjunction_layer.weights) + torch.sum(self.disjunction_layer.weights)

    def compute_l2_norm(self):
        return torch.sum(self.conjunction_layer.weights ** 2) + torch.sum(self.disjunction_layer.weights ** 2)

    def clip_weights(self):
        self.conjunction_layer.clip_weights()
        self.disjunction_layer.clip_weights()

    def get_rules(self, previous_layer, skip_connection_layer):
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
        self.conjunction_layer.forward_tot = self.disjunction_layer.forward_tot = self.forward_tot
        self.conjunction_layer.activation_nodes = self.disjunction_layer.activation_nodes = self.activation_nodes

    def get_rule_description(self, input_rule_name, wrap=False):
        self.rule_name = []
        for rules, operator in zip(self.rule_list, ('&', '|')):
            self._append_rule_descriptions(rules, input_rule_name, operator, wrap)

    def _append_rule_descriptions(self, rules, input_rule_name, operator, wrap):
        for rule in rules:
            description = self._build_rule_description(rule, input_rule_name, operator, wrap)
            self.rule_name.append(description)

    def _build_rule_description(self, rule, input_rule_name, operator, wrap):
        rule_description = ''
        for i, (layer_shift, rule_id) in enumerate(rule):
            prefix = f' {operator} ' if i != 0 else ''
            not_str = '~' if layer_shift > 0 else ''
            layer_shift *= -1 if layer_shift > 0 else 1
            var_name = input_rule_name[2 + layer_shift][rule_id]
            var_str = f'({var_name})' if wrap or not_str else var_name
            rule_description += f"{prefix}{not_str}{var_str}"
        return rule_description
