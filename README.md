# Moons Dataset Neural Network Classifier

A PyTorch implementation of a binary classification neural network trained on the scikit-learn moons dataset, with training/validation split and loss visualization.

## About

This project demonstrates a feedforward neural network for binary classification on the moons dataset - a classic toy dataset with two interleaving half circles. The implementation includes proper train/validation splitting, loss tracking, and accuracy metrics.

## Features

- Simple 2-layer neural network with ReLU activation
- Standardized input features using StandardScaler
- 80/20 train/validation split
- Real-time loss tracking for both training and validation
- Loss curve visualization
- Final validation accuracy reporting

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/Er1233/Dataset-model
cd moons dataset model lv2

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Training

Simply run the main script:

```bash
python main.py
```

The script will:
1. Generate the moons dataset (500 samples)
2. Split into train (400) and validation (100) sets
3. Train for 20 epochs
4. Display loss curves
5. Report final validation accuracy

### Output Example

```
==================================================
Moons Dataset Model
========================================
Training samples: 400
Validation samples: 100
parameters: 25

Training...
1 / 20
Avg_train_loss: 0.6234
Avg_val_loss: 0.5987
...
20 / 20
Avg_train_loss: 0.1523
Avg_val_loss: 0.1687

Final Validation Accuracy: 94.00%

Done!
```

## Model Architecture

```
Input Layer (2 features)
    ↓
Linear Layer (2 → 8 neurons)
    ↓
ReLU Activation
    ↓
Linear Layer (8 → 1 neuron)
    ↓
Sigmoid (via BCEWithLogitsLoss)
```

**Architecture Details:**
- **Input size:** 2 (x, y coordinates)
- **Hidden layer:** 8 neurons
- **Output:** 1 neuron (binary classification)
- **Activation:** ReLU
- **Output activation:** Sigmoid (implicit in loss function)
- **Total parameters:** 25 (trainable)

**Training Configuration:**
- **Loss function:** BCEWithLogitsLoss (Binary Cross Entropy with Logits)
- **Optimizer:** Adam
- **Learning rate:** 0.02
- **Batch size:** 32

## Dataset

**Moons Dataset** (generated with scikit-learn)
- **Total samples:** 500
- **Training samples:** 400 (80%)
- **Validation samples:** 100 (20%)
- **Classes:** 2 (binary classification)
- **Features:** 2D coordinates (x, y)
- **Noise level:** 0.2
- **Preprocessing:** StandardScaler normalization
- **Random seed:** 42 (for reproducibility)

## Code Structure

### Main Components

**1. AvdDataset Class**
- Generates moons dataset using `make_moons`
- Applies StandardScaler for feature normalization
- Implements PyTorch Dataset interface

**2. AdvModel Class**
- Simple 2-layer neural network
- 2 → 8 → 1 architecture
- ReLU activation between layers

**3. train_with_validation Function**
- Trains model for specified epochs
- Tracks both training and validation loss
- Returns loss history for visualization

**4. plot_with_validation Function**
- Creates matplotlib visualization
- Shows training vs validation loss curves
- Helps identify overfitting/underfitting

**5. main Function**
- Orchestrates the entire pipeline
- Handles data splitting
- Runs training and evaluation

## Results

### Training Performance

The model achieves excellent performance on the moons dataset:

| Metric | Value |
|--------|-------|
| Final Training Loss | ~0.15 |
| Final Validation Loss | ~0.17 |
| Validation Accuracy | ~94% |
| Training Time | < 1 minute (CPU) |
| Total Parameters | 25 |

### Loss Curves

The training produces a plot showing:
- **Blue line:** Training loss (decreases smoothly)
- **Orange line:** Validation loss (tracks training loss closely)
- **X-axis:** Epochs (1-20)
- **Y-axis:** Loss value

**Key Observations:**
- Both losses decrease steadily over epochs
- Minimal gap between training and validation loss
- No significant overfitting
- Model converges well within 20 epochs

## Project Structure

```
moons-classifier/
├── main.py              # Main training script (your code)
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Requirements

```
torch>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
numpy>=1.24.0
```

## Key Implementation Details

### Data Preprocessing
- Features are standardized using `StandardScaler` (mean=0, std=1)
- Ensures faster convergence and better training stability

### Train/Validation Split
- 80/20 split with fixed random seed (42)
- Ensures reproducible results across runs

### Loss Function Choice
- `BCEWithLogitsLoss` combines sigmoid activation and BCE loss
- More numerically stable than separate sigmoid + BCE
- Suitable for binary classification tasks

### Training Loop
- Model set to `.train()` mode during training
- Model set to `.eval()` mode during validation
- `torch.no_grad()` used for validation (no gradient computation)
- Batch-wise loss accumulation and averaging

## Customization

### Adjust Dataset Size
```python
dataset = AvdDataset(n_samples=1000)  # Change from 500
```

### Modify Architecture
```python
self.fc1 = nn.Linear(2, 16)  # Increase hidden units
self.fc2 = nn.Linear(16, 1)
```

### Change Training Parameters
```python
train_with_validation(model, train_loader, val_loader, epochs=50)  # More epochs
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate
```

### Adjust Batch Size
```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## Future Improvements

- [ ] Add early stopping based on validation loss
- [ ] Implement model checkpointing (save best model)
- [ ] Add decision boundary visualization
- [ ] Experiment with deeper architectures
- [ ] Add learning rate scheduling
- [ ] Include confusion matrix
- [ ] Add more evaluation metrics (precision, recall, F1)
- [ ] Support for GPU training
- [ ] Command-line arguments for hyperparameters

## Troubleshooting

**Issue:** Loss not decreasing
- Try lower learning rate (e.g., 0.001)
- Increase number of epochs
- Check data normalization

**Issue:** Overfitting (validation loss increases)
- Add dropout layers
- Reduce model capacity
- Increase dataset size

**Issue:** Poor accuracy
- Increase hidden layer size
- Train for more epochs
- Adjust noise level in dataset

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset from scikit-learn's `make_moons` function
- Built with PyTorch deep learning framework
- Inspired by classic machine learning tutorials

## Contact

Erick muteti - [email@ mutetie56@gmail.com](mutetie56@gmail.com)

Project Link:https://github.com/Er1233/Dataset-model

---

**Note:** This is an educational project demonstrating basic neural network concepts with PyTorch. The moons dataset is a simple toy problem ideal for learning and experimentation.