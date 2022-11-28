import torch
import torchvision

import model_builder
import argparse

# Create image and model_pth arguments for our predict script
parser = argparse.ArgumentParser()
parser.add_argument("--image",
                    help="target_image filepath to predict on")

parser.add_argument("--model_pth",
                    default="models/05_going_modular_script_mode_tinyvgg_model.pth",
                    help="target_model to usr for prediction")


args = parser.parse_args()

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# SEt image path
image_path = args.image
print(f"[INFO] Predicting on {image_path}")

# Function to load in the model
def load_model(filepath=args.model_pth):
    # Need to use same hyperparameters as saved model
    model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=128,
                                  output_shape=3).to(device)

    print(f"[INFO] Loading in model from: {filepath}")
    # Load in the saved model state dictionary from file
    model.load_state_dict(torch.load(filepath))
    return model

# Function to load in model + predict on select image
def predict_on_image(image_path=image_path, filepath=args.model_pth):
    # Load the model
    model = load_model(filepath)

    # Load in the image and turn it into torch.float32 (same type as model)
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Preprocess the image to get it between 0 and 1
    image = image / 255.

    # Resize the image to be the same size as the model
    transform = torchvision.transforms.Resize(size=(64, 64))
    image = transform(image)

    # Predict on image
    model.eval()
    with torch.inference_mode():
        # Put image to target device
        image = image.to(device)

        # Get pred logits
        pred_logits = model(image.unsqueeze(dim=0)) # make sure image has batch dimension (shape: [batch_size, height, width, color_channels])

        # Get pred probs
        pred_prob = torch.softmax(pred_logits, dim=1)

        # Get pred labels
        pred_label = torch.argmax(pred_prob, dim=1)
        pred_label_class = class_names[pred_label]

    print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
    predict_on_image()
