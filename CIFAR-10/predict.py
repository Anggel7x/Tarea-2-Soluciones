import requests
import numpy as np
from PIL import Image
from io import BytesIO
from keras.models import load_model


def read_image_from_url(url: str) -> Image:
    try:
        # Make a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Read the image data from the response content
            image_data = response.content

            # Create a BytesIO object to handle the image data
            image_buffer = BytesIO(image_data)

            # Open the image using PIL (Pillow)
            image = Image.open(image_buffer)

            # You can now work with the 'image' object (e.g., display or save it)
            return image
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def predict_img(img_url: str) -> str:
    image = read_image_from_url(url=img_url)
    image = image.resize((32, 32))

    image_arr = np.array(image)
    image_arr = image_arr.reshape((32, 32, 3))
    image_arr = image_arr / 255.0

    recon_img_model = load_model(filepath="trained_models/vgg_best.h5")

    prediction = recon_img_model.predict(np.expand_dims(image_arr, axis=0), verbose=0)

    map = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    index = np.argmax(prediction)
    label = map[index]

    return label
