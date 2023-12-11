from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained Keras model (replace 'model_path' with the actual path to your saved model)
model = tf.keras.models.load_model("./my_model")


# Define a route for the default page
@app.route("/")
def index():
    # Render the upload HTML form
    return render_template("./upload.html")


# Define a route for the action of the form, for example '/predict/'
@app.route("/predict/", methods=["POST"])
def predict():
    if request.method == "POST":
        uploaded_files = request.files.getlist("file[]")  # Notice the change here
        print("Uploaded files received:", uploaded_files)  # Debugging line

        predictions = []
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            print("Current file being processed:", filename)  # Debugging line

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join("static", "uploads", filename)
            print("File path where image will be saved:", file_path)  # Debugging line
            file.save(file_path)

            # Open the image file and preprocess it
            img = Image.open(file_path)
            img = img.resize((160, 160))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis

            # Predict the class
            prediction = model.predict(img_array)
            probability = tf.nn.sigmoid(prediction).numpy()[0][0]
            predicted_class = "Sunflower" if probability > 0.5 else "Cauliflower"
            probability_percent = (
                round(probability * 100, 2)
                if predicted_class == "Sunflowers"
                else round((1 - probability) * 100, 2)
            )

            predictions.append(
                {
                    "filename": file.filename,
                    "class": predicted_class,
                    "probability": probability_percent,
                }
            )

        # Separate the images by class
        sunflowers = [p for p in predictions if p["class"] == "Sunflower"]
        cauliflowers = [p for p in predictions if p["class"] == "Cauliflower"]

        # Return the result as JSON
        return render_template("results.html", sunflowers=sunflowers, cauliflowers=cauliflowers)


if __name__ == "__main__":
    app.run(debug=True)
