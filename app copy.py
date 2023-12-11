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
        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)

        # Open the image file
        img = Image.open(file_path)

        # Preprocess the image
        img = img.resize((160, 160))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Predict the class
        predictions = model.predict(img_array)
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)

        # Determine the class label
        class_names = ["Cauliflower", "Sunflower"]  # Update class names as necessary
        predicted_class = class_names[int(predictions.numpy()[0][0])]

        # Return the result as JSON
        return render_template("./result.html", prediction=predicted_class)


if __name__ == "__main__":
    app.run(debug=True)
