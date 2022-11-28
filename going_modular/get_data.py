
import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = "data/"
image_path = data_path / "pizza_steak_sushi"

# If th image folder doesn't exist, download it or else skip it
if image_path.is_dir():
    print(f"{image_path} already exists, skipping download")

else:
    # Download pizza, steak and sushi
    with open(image_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print(f"Downloading data....")
        f.write(request.content)

    # Unzip pizza, steak data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip")as zip_ref:
        print(f"Unzipping pizza, steak, sushi...")
        zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path/ "pizza_steak_sushi.zip")
