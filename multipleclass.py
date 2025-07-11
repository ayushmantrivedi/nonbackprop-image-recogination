import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import deque
import random
from concurrent.futures import ThreadPoolExecutor
from numba import njit

# Hyperparameters
LEVEL1_NEURONS = 50
LEVEL2_NEURONS = 20
POP_SIZE = 20  # Increased from 7 to 20 for better evolutionary stability
VM_HISTORY = 20
VM_INFLUENCE_PROB = 0.2
VM_IMPROVEMENT_THRESH = 0.15  # 15% improvement
TAU1 = 0.15
TAU2 = 0.10
MUT_STRENGTH_BASE = 0.1
EPOCHS = 50
PRINT_INTERVAL = 5
THOUGHT_INTERVAL = 10
MIN_MUT_STRENGTH = 0.01  # Minimum mutation strength to avoid over-exploration

np.random.seed(42)
random.seed(42)

# Load and preprocess data
digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target
num_classes = len(np.unique(y))

# Normalize X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode y
try:
    ohe = OneHotEncoder(sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(sparse=False)
y_onehot = ohe.fit_transform(y.reshape(-1, 1))

# Train/test split
X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = train_test_split(
    X, y, y_onehot, test_size=0.2, random_state=42, stratify=y)

# --- Evolutionary Neuron ---
@njit
def population_forward(x, weights, biases):
    # Ensure all are float32 for Numba compatibility
    return np.dot(weights.astype(np.float32), x.astype(np.float32)) + biases.astype(np.float32)

class EvoNeuron:
    def __init__(self, input_dim, pop_size=POP_SIZE):
        self.input_dim = input_dim
        self.pop_size = pop_size
        self.population = [self._random_individual() for _ in range(pop_size)]
        self.best_idx = 0
        self.last_error = None
        self.last_output = None
        self.last_weights = None
        self.last_bias = None
        self.elite = None  # Store the best individual ever found
        self.elite_error = float('inf')

    def _random_individual(self):
        return {
            'weights': np.random.randn(self.input_dim).astype(np.float32),
            'bias': np.float32(np.random.randn()),
        }

    def _check_population_shapes(self):
        # [DIMENSION CHECK] Ensure each individual's weights match the expected input dimension for this neuron.
        # If not, reinitialize the population to correct the shape mismatch.
        for ind in self.population:
            if ind['weights'].shape != (self.input_dim,):
                print(f"[DEBUG] Population shape mismatch detected. Reinitializing population for input_dim={self.input_dim}.")
                self.population = [self._random_individual() for _ in range(self.pop_size)]
                break

    def forward(self, x, y_true, error_fn, mutation_strength, V_m=None):
        self._check_population_shapes()
        # Vectorized population evaluation
        weights = np.array([ind['weights'] for ind in self.population])
        biases = np.array([ind['bias'] for ind in self.population])
        outs = population_forward(x, weights, biases)
        errors = np.array([error_fn(out, y_true) for out in outs])
        best_idx = np.argmin(errors)
        self.best_idx = best_idx
        self.last_error = errors[best_idx]
        self.last_output = outs[best_idx]
        self.last_weights = self.population[best_idx]['weights'].copy()
        self.last_bias = self.population[best_idx]['bias']
        # Elitism: update elite if this is the best ever
        if self.last_error < self.elite_error:
            self.elite_error = self.last_error
            self.elite = {
                'weights': self.last_weights.copy(),
                'bias': self.last_bias
            }
        return self.last_output, self.last_error

    def _activate(self, x):
        # Simple linear for hidden, softmax for output handled outside
        return x

    def evolve(self, x, y_true, error_fn, mutation_strength, V_m=None):
        self._check_population_shapes()
        # Selection: keep top 2
        errors = []
        for ind in self.population:
            out = self._activate(np.dot(x, ind['weights']) + ind['bias'])
            err = error_fn(out, y_true)
            errors.append(err)
        idx_sorted = np.argsort(errors)
        survivors = [self.population[i] for i in idx_sorted[:2]]
        # Elitism: always keep the best individual ever found
        if self.elite is not None:
            survivors.append({'weights': self.elite['weights'].copy(), 'bias': self.elite['bias']})
        # Mutation: create new individuals
        new_pop = survivors.copy()
        while len(new_pop) < self.pop_size:
            parent = random.choice(survivors)
            assert parent['weights'].shape == (self.input_dim,), f"Parent weights shape {parent['weights'].shape} != {self.input_dim}"  # [DIMENSION ASSERTION] Ensure mutation uses correct shape
            child = {
                'weights': parent['weights'] + np.random.randn(self.input_dim) * mutation_strength,
                'bias': parent['bias'] + np.random.randn() * mutation_strength
            }
            # V_m influence
            if V_m and len(V_m) > 0 and random.random() < VM_INFLUENCE_PROB:
                v = random.choice(V_m)
                if v['weights'].shape == (self.input_dim,):
                    child['weights'] += v['weights'] * 0.5  # scale influence
                child['bias'] += v['bias'] * 0.5
            new_pop.append(child)
        self.population = new_pop

# --- Level 3 Output Neuron (Softmax) ---
class OutputNeuron(EvoNeuron):
    def _activate(self, x):
        return x  # Linear, softmax applied at layer

# --- Utility Functions ---
def mse_loss(pred, true):
    return np.mean((pred - true) ** 2)

def ce_loss(pred, true):
    # pred: (num_classes,), true: (num_classes,)
    pred = np.clip(pred, 1e-8, 1-1e-8)
    return -np.sum(true * np.log(pred))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def ce_loss_with_confidence(pred, true, reward_weight=0.1):
    # pred: (num_classes,), true: (num_classes,)
    pred = np.clip(pred, 1e-8, 1-1e-8)
    ce = -np.sum(true * np.log(pred))
    # Confidence reward: encourage the correct class probability to be much higher than the max of others
    correct_prob = np.sum(pred * true)
    max_other_prob = np.max(pred * (1 - true))
    confidence_margin = correct_prob - max_other_prob
    reward = reward_weight * confidence_margin
    return ce - reward  # Subtract reward to lower loss when confidence is high

# --- Significant Mutation Vector (V_m) ---
class SignificantMutationVector:
    def __init__(self, maxlen=VM_HISTORY):
        self.deque = deque(maxlen=maxlen)
    def add(self, weights, bias):
        self.deque.append({'weights': weights.copy(), 'bias': bias})
    def get(self):
        return list(self.deque)

# --- Model ---
class MultiClassEvoNet:
    def __init__(self, input_dim, num_classes):
        self.level1 = [EvoNeuron(input_dim) for _ in range(LEVEL1_NEURONS)]
        self.level2 = [EvoNeuron(LEVEL1_NEURONS) for _ in range(LEVEL2_NEURONS)]
        self.level3 = [OutputNeuron(LEVEL2_NEURONS) for _ in range(num_classes)]
        self.num_classes = num_classes
        self.V_m = SignificantMutationVector()
        self.tau1 = TAU1
        self.tau2 = TAU2
        self.mut_strength_base = MUT_STRENGTH_BASE
        self.global_error = 1.0

    def get_mutation_strength(self):
        # More aggressive decay: square the global error
        mut_strength = self.mut_strength_base * (self.global_error ** 2)
        return max(mut_strength, MIN_MUT_STRENGTH)

    def forward(self, x, y_true, train=True):
        mut_strength = self.get_mutation_strength()
        # Level 1 (sequential for speed)
        l1_outputs = []
        l1_errors = []
        l1_marks = []
        for neuron in self.level1:
            out, err = neuron.forward(x, y_true, mse_loss, mut_strength, self.V_m.get())
            l1_outputs.append(out)
            l1_errors.append(err)
            if err < self.tau1:
                l1_marks.append(out)
            else:
                l1_marks.append(('*', out))  # Mark failed neuron output as tuple
        l1_outputs = np.array(l1_outputs, dtype=object)
        l1_errors = np.array(l1_errors)
        # Level 1: pass all outputs, but mark failed ones
        l2_inputs = l1_marks
        assert len(l2_inputs) == LEVEL1_NEURONS  # [DIMENSION ASSERTION] Ensure correct input size for Level 2
        # Level 2 (sequential for speed)
        l2_outputs = []
        l2_errors = []
        l2_marks = []
        for neuron, inp in zip(self.level2, l2_inputs):
            # If input is marked, extract value for computation
            if isinstance(inp, tuple) and inp[0] == '*':
                inp_val = inp[1]
            else:
                inp_val = inp
            out, err = neuron.forward(np.full(LEVEL1_NEURONS, inp_val), y_true, mse_loss, mut_strength, self.V_m.get())
            l2_outputs.append(out)
            l2_errors.append(err)
            if err < self.tau2:
                l2_marks.append(out)
            else:
                l2_marks.append(('*', out))  # Mark failed neuron output as tuple
        l2_outputs = np.array(l2_outputs, dtype=object)
        l2_errors = np.array(l2_errors)
        # Level 2: pass all outputs, but mark failed ones
        l3_inputs = l2_marks
        assert len(l3_inputs) == LEVEL2_NEURONS  # [DIMENSION ASSERTION] Ensure correct input size for Level 3
        # Level 3 (sequential for speed)
        l3_outputs = []
        for i, (neuron, inp) in enumerate(zip(self.level3, l3_inputs)):
            if isinstance(inp, tuple) and inp[0] == '*':
                inp_val = inp[1]
            else:
                inp_val = inp
            out, _ = neuron.forward(np.full(LEVEL2_NEURONS, inp_val), y_true[i], mse_loss, mut_strength, self.V_m.get())
            l3_outputs.append(out)
        l3_outputs = np.array(l3_outputs)
        y_pred = softmax(l3_outputs)
        # Level 3: evolve (sequential)
        if train:
            for i, neuron in enumerate(self.level3):
                inp = l3_inputs[i]
                inp_val = inp[1] if isinstance(inp, tuple) and inp[0] == '*' else inp
                neuron.evolve(np.full(LEVEL2_NEURONS, inp_val), y_true[i], mse_loss, mut_strength, self.V_m.get())
        return y_pred, l1_errors, l2_errors, l3_outputs

    def train(self, X, y, y_oh, epochs=EPOCHS, X_val=None, y_val=None, y_val_oh=None):
        for epoch in range(1, epochs+1):
            print(f"Starting epoch {epoch}")
            correct = 0
            total_loss = 0
            for i in range(X.shape[0]):
                x = X[i]
                y_true = y_oh[i]
                y_label = y[i]
                y_pred, l1_errs, l2_errs, l3_outs = self.forward(x, y_true, train=True)
                pred_label = np.argmax(y_pred)
                if pred_label == y_label:
                    correct += 1
                loss = ce_loss_with_confidence(y_pred, y_true)
                total_loss += loss
                # --- Benchmark neuron and V_m update ---
                # Level 1
                l1_bench_idx = np.argmin(l1_errs)
                l1_bench_err = l1_errs[l1_bench_idx]
                l1_bench = self.level1[l1_bench_idx]
                if l1_bench.last_error is not None and l1_bench.last_error < self.tau1:
                    self.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
                # Level 2
                l2_bench_idx = np.argmin(l2_errs)
                l2_bench_err = l2_errs[l2_bench_idx]
                l2_bench = self.level2[l2_bench_idx]
                if l2_bench.last_error is not None and l2_bench.last_error < self.tau2:
                    self.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
                # Level 3
                # (Not updating V_m for output layer, but could be added)
            acc = correct / X.shape[0]
            avg_loss = total_loss / X.shape[0]
            self.global_error = avg_loss  # Use for mutation scaling
            print(f"Finished epoch {epoch}")
            if epoch % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch}: Global Accuracy: {acc*100:.2f}%, Loss: {avg_loss:.4f}")
            if epoch % THOUGHT_INTERVAL == 0:
                print(f"[Thought] Epoch {epoch}: Global error is {self.global_error:.4f}, mutation strength now {self.get_mutation_strength():.4f}")
            # Optionally, evaluate on validation set
            if X_val is not None and epoch % PRINT_INTERVAL == 0:
                val_acc, val_loss = self.evaluate(X_val, y_val, y_val_oh)
                print(f"[Validation] Accuracy: {val_acc*100:.2f}%, Loss: {val_loss:.4f}")

    def evaluate(self, X, y, y_oh):
        correct = 0
        total_loss = 0
        for i in range(X.shape[0]):
            x = X[i]
            y_true = y_oh[i]
            y_label = y[i]
            y_pred, _, _, _ = self.forward(x, y_true, train=False)
            pred_label = np.argmax(y_pred)
            if pred_label == y_label:
                correct += 1
            loss = ce_loss(y_pred, y_true)
            total_loss += loss
        acc = correct / X.shape[0]
        avg_loss = total_loss / X.shape[0]
        return acc, avg_loss

# --- Main ---
if __name__ == "__main__":
    model = MultiClassEvoNet(input_dim=X_train.shape[1], num_classes=num_classes)
    model.train(X_train, y_train, y_train_oh, epochs=EPOCHS, X_val=X_test, y_val=y_test, y_val_oh=y_test_oh)
    test_acc, test_loss = model.evaluate(X_test, y_test, y_test_oh)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%, Loss: {test_loss:.4f}")
