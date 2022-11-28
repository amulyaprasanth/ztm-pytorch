"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import argparse
import torch
import  data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hyperparameters
parser = argparse.ArgumentParser(description="Getting some hyperparmeters.")

parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs for the model to train")
parser.add_argument("--hidden_units", type=int, default=10, help="number of hidden units in hidden_layers")
parser.add_argument("--batch_size", type=int, default=32, help="batch size for our dataloaders")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for our optimizer")

parser.add_argument("--train_dir", type=str, default="data/pizza_steak_sushi/train", help="directory file path containing training data in standard image classification format")
parser.add_argument("--test_dir", type=str, default="data/pizza_steak_sushi/test", help="directory file path containing test data in standard image classification format")

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="05_going_modular_script_mode_tinyvgg_model.pth")
