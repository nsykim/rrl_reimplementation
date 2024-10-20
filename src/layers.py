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

def stochastic_product(inputs):
    """A stochastic product function that adds noise during training."""
    noise = torch.rand_like(inputs) * 0.1
    return torch.prod(inputs + noise, dim=1)

def standard_product(inputs):
    """Standard product function."""
    return torch.prod(inputs, dim=1)

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
        self.discrete_feature_count = input_shape[0] * (2 if use_negation else 1)
        self.continuous_feature_count = input_shape[1]
        self.output_size = self.discrete_feature_count + 2 * num_bins * self.continuous_feature_count
        self.feature_mapping = {i: i for i in range(self.output_size)}

        self.min_val = nn.Parameter(min_val, requires_grad=False) if min_val is not None else None
        self.max_val = nn.Parameter(max_val, requires_grad=False) if max_val is not None else None

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
    def discretize(self, input_data):
        return self.forward(input_data)

    def enforce_bounds(self):
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
    def __init__(self, num_outputs, input_dim, use_negation=False, stochastic_grad=False):
        super(LinearRegressionLayer, self).__init__()
        self.num_outputs = num_outputs
        self.use_negation = use_negation
        self.input_dim = input_dim * (2 if use_negation else 1)
        self.output_dim = num_outputs
        self.layer_type = 'linear_regression'
        
        # Linear transformation layer
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.product_function = stochastic_product if stochastic_grad else standard_product

    def forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)
        return self.linear(inputs)

    @torch.no_grad()
    def binarized_forward(self, inputs):
        return self.forward(inputs)

    def enforce_weight_clipping(self, lower_bound=-1.0, upper_bound=1.0):
        with torch.no_grad():
            for param in self.linear.parameters():
                param.data.clamp_(lower_bound, upper_bound)

    def compute_l1_norm(self):
        return torch.norm(self.linear.weight, p=1)
    
    def compute_l2_norm(self):
        return torch.sum(self.linear.weight ** 2)
    
    def calculate_rule_weights(self, previous_layer, skip_connection_layer=None):
        prev_activation_count = previous_layer.node_activation_count
        prev_forward_count = previous_layer.forward_total
        always_active_pos = prev_activation_count == prev_forward_count
        
        prev_dim_map = {idx: (-1, dim) for idx, dim in previous_layer.node_to_rule_map.items()}
        
        if skip_connection_layer is not None:
            skip_shifted_dim_map = {(idx + previous_layer.output_dim): (-2, dim)
                                    for idx, dim in skip_connection_layer.node_to_rule_map.items()}
            merged_dim_map = {**skip_shifted_dim_map, **prev_dim_map}
            always_active_pos = torch.cat(
                [always_active_pos, 
                 skip_connection_layer.node_activation_count == skip_connection_layer.forward_total]
            )
        else:
            merged_dim_map = prev_dim_map
        
        weight_matrix, bias_vector = self.linear.weight.detach(), self.linear.bias.detach()
        bias_vector = torch.sum(weight_matrix.T[always_active_pos], dim=0) + bias_vector
        weight_matrix = weight_matrix.cpu().numpy()
        self.bias_vector = bias_vector.cpu().numpy()

        rule_weight_map = defaultdict(lambda: defaultdict(float))
        rule_to_dim_map = {}

        for label_idx, weights in enumerate(weight_matrix):
            for dim_idx, weight_value in enumerate(weights):
                rule_id = merged_dim_map.get(dim_idx, (-1, -1))
                if rule_id[1] == -1:
                    continue
                rule_weight_map[rule_id][label_idx] += weight_value
                rule_to_dim_map[rule_id] = dim_idx % previous_layer.output_dim
        
        self.rule_to_dim_map = rule_to_dim_map
        self.rule_to_weight_map = sorted(
            rule_weight_map.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True
        )

class ConjunctionLayer(nn.Module):
    def __init__(self, num_conjunctions, input_dim, use_negation=False, alpha=0.999, beta=8, gamma=1, stochastic_grad=False):
        super(ConjunctionLayer, self).__init__()
        self.num_conjunctions = num_conjunctions
        self.use_negation = use_negation
        self.input_dim = input_dim * (2 if use_negation else 1)
        self.output_dim = num_conjunctions
        self.layer_type = 'logical_conjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))
        self.product_function = stochastic_product if stochastic_grad else standard_product

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs):
        continuous_output = self.continuous_logic(inputs)
        binarized_output = self.binarized_logic(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_logic(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)

        x_transform = 1 - torch.sigmoid(inputs * self.alpha)
        weight_transform = 1 - torch.sigmoid(self.weights * self.alpha)

        return torch.pow(1.0 / (1.0 + torch.matmul(x_transform, weight_transform)), self.gamma)

    @torch.no_grad()
    def binarized_logic(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)

        binary_weights = Binarize.apply(self.weights - THRESHOLD)
        return torch.where(torch.matmul(1 - inputs, binary_weights) > 0,
                           torch.zeros_like(inputs), torch.ones_like(inputs))

    def clip_weights(self):
        self.weights.data.clamp_(INIT_L, 1.0)

class DisjunctionLayer(nn.Module):
    def __init__(self, num_disjunctions, input_dim, use_negation=False, alpha=0.999, beta=8, gamma=1, stochastic_grad=False):
        super(DisjunctionLayer, self).__init__()
        self.num_disjunctions = num_disjunctions
        self.use_negation = use_negation
        self.input_dim = input_dim * (2 if use_negation else 1)
        self.output_dim = num_disjunctions
        self.layer_type = 'disjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_dim, self.num_disjunctions))
        self.product_function = stochastic_product if stochastic_grad else standard_product

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs):
        continuous_output = self.continuous_disjunction(inputs)
        binarized_output = self.binarized_disjunction(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_disjunction(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)

        input_transform = torch.sigmoid(inputs * self.alpha)
        weight_transform = torch.sigmoid(self.weights * self.alpha)
        result = torch.pow(1.0 / (1.0 + torch.matmul(input_transform, weight_transform)), self.gamma)

        return result

    @torch.no_grad()
    def binarized_disjunction(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)

        binary_weights = Binarize.apply(self.weights - THRESHOLD)
        result = torch.matmul(inputs, binary_weights)
        return torch.where(result > 0, torch.ones_like(result), torch.zeros_like(result))

    def clip_weights(self):
        self.weights.data.clamp_(INIT_L, 1.0)

class OriginalConjunctionLayer(nn.Module):
    def __init__(self, n, input_dim, use_negation=False, stochastic_grad=False):
        super(OriginalConjunctionLayer, self).__init__()
        self.n = n
        self.use_negation = use_negation
        self.input_dim = input_dim * (2 if use_negation else 1)
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_dim, self.n))
        self.product_function = stochastic_product if stochastic_grad else standard_product

    def forward(self, inputs):
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, x):
        x = augment_with_negation(x, self.use_negation)
        return self.product_function(1 - (1 - x).unsqueeze(-1) * self.weights)

    @torch.no_grad()
    def binarized_forward(self, x):
        x = augment_with_negation(x, self.use_negation)
        binarized_weights = Binarize.apply(self.weights - THRESHOLD)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * binarized_weights, dim=1)

    def clip(self):
        self.weights.data.clamp_(0.0, 1.0)

class OriginalDisjunctionLayer(nn.Module):
    def __init__(self, n, input_dim, use_negation=False, stochastic_grad=False):
        super(OriginalDisjunctionLayer, self).__init__()
        self.n = n
        self.use_negation = use_negation
        self.input_dim = input_dim * (2 if use_negation else 1)
        self.output_dim = n
        self.layer_type = 'disjunction'

        self.weights = nn.Parameter(0.5 * torch.rand(self.input_dim, n))
        self.product_function = stochastic_product if stochastic_grad else standard_product

    def forward(self, inputs):
        continuous_output = self.continuous_forward(inputs)
        binarized_output = self.binarized_forward(inputs)
        return GradientGraft.apply(binarized_output, continuous_output)

    def continuous_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)
        return 1 - self.product_function(1 - inputs)

    @torch.no_grad()
    def binarized_forward(self, inputs):
        inputs = augment_with_negation(inputs, self.use_negation)
        binarized_weights = (self.weights > 0.5).float()
        return 1 - torch.prod(1 - inputs * binarized_weights, dim=1)

    def clip(self):
        self.weights.data.clamp_(0.0, 1.0)


def extract_rules(previous_layer, skip_connection_layer, current_layer, position_shift=0):
    node_to_rule_map = defaultdict(lambda: -1)
    rules = {}
    rule_counter = 0
    rule_list = []

    # binarized_weights shape = (number_of_nodes, input_dimensions)
    binarized_weights = (current_layer.W.t() > 0.5).type(torch.int).detach().cpu().numpy()

    # Merge node_to_rule_map from the previous layer and skip connection layer (if available)
    merged_node_map = previous_node_map = {key: (-1, val) for key, val in previous_layer.node_to_rule_map.items()}
    if skip_connection_layer is not None:
        shifted_node_map = {
            (key + previous_layer.output_dim): (-2, val) 
            for key, val in skip_connection_layer.node_to_rule_map.items()
        }
        merged_node_map = defaultdict(lambda: -1, {**shifted_node_map, **previous_node_map})

    for node_index, weights_row in enumerate(binarized_weights):
        # Skip nodes that are inactive (dead nodes) or fully active (always triggered)
        if current_layer.node_activation_cnt[node_index + position_shift] == 0 or \
           current_layer.node_activation_cnt[node_index + position_shift] == current_layer.forward_tot:
            node_to_rule_map[node_index + position_shift] = -1
            continue
        
        rule = {}
        feature_bounds = {}

        # Special handling for binarization layers to account for discrete features
        if previous_layer.layer_type == 'binarization' and previous_layer.input_dim[1] > 0:
            discrete_features = torch.cat((previous_layer.cl.t().reshape(-1), previous_layer.cl.t().reshape(-1))).detach().cpu().numpy()

        for weight_index, weight in enumerate(weights_row):
            # Handling for 'NOT' logic in input dimensions
            negate_input = 1
            if current_layer.use_not:
                if weight_index >= current_layer.input_dim // 2:
                    negate_input = -1
                weight_index = weight_index % (current_layer.input_dim // 2)

            # Only consider positive weights and valid previous mappings
            if weight > 0 and merged_node_map[weight_index][1] != -1:
                if previous_layer.layer_type == 'binarization' and weight_index >= previous_layer.disc_num:
                    # Handle discrete input features
                    feature_index = weight_index - previous_layer.disc_num
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
        
        # Assign unique rule IDs and store in node_to_rule_map
        sorted_rule = tuple(sorted(rule.keys()))
        if sorted_rule not in rules:
            rules[sorted_rule] = rule_counter
            rule_list.append(sorted_rule)
            node_to_rule_map[node_index + position_shift] = rule_counter
            rule_counter += 1
        else:
            node_to_rule_map[node_index + position_shift] = rules[sorted_rule]

    return node_to_rule_map, rule_list

class UnionLayer(nn.Module):
    def __init__(self, num_units, input_dim, use_negation=False, use_novel_activation=False, estimated_grad=False, alpha=0.999, beta=8, gamma=1):
        super(UnionLayer, self).__init__()
        self.num_units = num_units
        self.use_negation = use_negation
        self.input_dim = input_dim
        self.output_dim = self.num_units * 2  # Union of conjunction and disjunction
        self.layer_type = 'union'
        
        if use_novel_activation:
            self.conjunction_layer = ConjunctionLayer(num_conjunctions=self.num_units, input_dim=self.input_dim, use_negation=self.use_negation, alpha=alpha, beta=beta, gamma=gamma, stochastic_grad=estimated_grad)
            self.disjunction_layer = DisjunctionLayer(num_conjunctions=self.num_units, input_dim=self.input_dim, use_negation=self.use_negation, alpha=alpha, beta=beta, gamma=gamma, stochastic_grad=estimated_grad)
        else:
            self.disjunction_layer = OriginalConjunctionLayer(n=self.num_units, input_dim=self.input_dim, use_negation=self.use_negation, stochastic_grad=estimated_grad)
            self.disjunction_layer = OriginalDisjunctionLayer(n=self.num_units, input_dim=self.input_dim, use_negation=self.use_negation, stochastic_grad=estimated_grad)

    def forward(self, input_tensor):
        conjunction_output = self.conjunction_layer(input_tensor)
        disjunction_output = self.disjunction_layer(input_tensor)
        return GradientGraft.apply(conjunction_output, disjunction_output)

    def binarized_forward(self, input_tensor):
        conjunction_output = self.conjunction_layer.binarized_forward(input_tensor)
        disjunction_output = self.disjunction_layer.binarized_forward(input_tensor)
        return torch.cat([conjunction_output, disjunction_output], dim=1)

    def edge_count(self):
        return self._sum_binarized_weights(self.conjunction_layer) + self._sum_binarized_weights(self.disjunction_layer)

    def _sum_binarized_weights(self, layer):
        binarized_weights = Binarize.apply(layer.weights - THRESHOLD)
        return torch.sum(binarized_weights)

    def l1_norm(self):
        return torch.sum(self.conjunction_layer.weights) + torch.sum(self.disjunction_layer.weights)

    def l2_norm(self):
        return torch.sum(self.conjunction_layer.weights ** 2) + torch.sum(self.disjunction_layer.weights ** 2)

    def clip_weights(self):
        self.conjunction_layer.clip_weights()
        self.disjunction_layer.clip_weights()

    def get_rules(self, previous_layer, skip_connection_layer):
        self._sync_layer_stats()

        # Extract rules from conjunction and disjunction layers
        conjunction_rules, conjunction_rule_list = self._extract_rules_for_layer(self.conjunction_layer, previous_layer, skip_connection_layer)
        disjunction_rules, disjunction_rule_list = self._extract_rules_for_layer(self.disjunction_layer, previous_layer, skip_connection_layer, shift=len(conjunction_rules))

        # Combine rules from both layers
        self.dim2id = defaultdict(lambda: -1, {**conjunction_rules, **disjunction_rules})
        self.rule_list = (conjunction_rule_list, disjunction_rule_list)

    def _sync_layer_stats(self):
        self.conjunction_layer.forward_tot = self.disjunction_layer.forward_tot = self.forward_tot
        self.conjunction_layer.node_activation_cnt = self.disjunction_layer.node_activation_cnt = self.node_activation_cnt

    def _extract_rules_for_layer(self, layer, previous_layer, skip_connection_layer, shift=0):
        rule_dim2id, rule_list = extract_rules(previous_layer, skip_connection_layer, layer, position_shift=shift)
        return rule_dim2id, rule_list

    def get_rule_description(self, input_rule_name, wrap_logic=False):
        self.rule_name = []
        for rule_list, operator in zip(self.rule_list, ('&', '|')):
            self._append_rule_descriptions(rule_list, input_rule_name, operator, wrap_logic)

    def _append_rule_descriptions(self, rule_list, input_rule_name, operator, wrap_logic):
        for rule in rule_list:
            description = self._build_rule_description(rule, input_rule_name, operator, wrap_logic)
            self.rule_name.append(description)

    def _build_rule_description(self, rule, input_rule_name, operator, wrap_logic):
        rule_description = ''
        for i, (layer_shift, rule_id) in enumerate(rule):
            prefix = f' {operator} ' if i != 0 else ''
            not_str = '~' if layer_shift > 0 else ''
            layer_shift *= -1 if layer_shift > 0 else 1
            var_name = input_rule_name[2 + layer_shift][rule_id]
            var_str = f'({var_name})' if wrap_logic or not_str else var_name
            rule_description += f"{prefix}{not_str}{var_str}"
        return rule_description
