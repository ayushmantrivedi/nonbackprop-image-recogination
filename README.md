# nonbackprop-image-recogination
# Evolutionary Neural Network (Non-Backprop) for Multiclass Classification

This project implements a population-based, evolutionary neural network for multiclass classification **without backpropagation**. It uses evolutionary strategies, significant mutation vectors, elitism, and a confidence reward to achieve high accuracy and encourage confident predictions.

## Features

- No gradient descent or backpropagation
- Population-based evolutionary learning
- Elitism: always keeps the best solution found
- Confidence reward in the loss function
- Tested on the scikit-learn digits dataset (1797 samples, 64 features, 10 classes)
- High accuracy (99%+) achievable

## How It Works

- Each neuron is a population of candidate solutions (weights/biases)
- Evolutionary selection, mutation, and knowledge sharing (via significant mutation vectors)
- Confidence reward encourages the model to make more confident predictions
- Elitism ensures the best solution is never lost

## Usage

1. **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the model:**
    ```bash
    python multipleclass.py
    ```

3. **(Optional) Modify hyperparameters in `multipleclass.py` to experiment.**

## Results

- Achieves >99% accuracy on the digits dataset
- Confidence reward helps lower loss and improve probability calibration

## Example Output

```
Epoch 5: Global Accuracy: 99.23%, Loss: 1.6542
[Validation] Accuracy: 100.00%, Loss: 1.6197
...
Final Test Accuracy: 99.44%, Loss: 1.6229
```

## Contributing

Pull requests and suggestions are welcome! Try new datasets, tweak the evolutionary logic, or add visualizations.

## License

[MIT License](LICENSE)
