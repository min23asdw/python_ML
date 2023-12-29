from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import io
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app)

# Load your Keras model
model = tf.keras.models.load_model('model/model_forAPP.h5', compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["categorical_accuracy"],
)

# Class labels for your model
class_labels = ['dewberry_blue', 'creco','Ahh','rollercoaster_cheese','bengbeng','bento','deno_stone','chocopie','rollercoaster_spicy','kitkat','lays_3','lays_cheese','lays_original','dewberry_red','lay_green','ff','oreo','pringles_green','lotus','marujo_red','marujo_green','tilli_indigo','tasto_spicy','tasto_honey','snackjack_chicken','snackjack_saltpepper','tawan_Larb','snackjack_shell','tilli_blue','tilli_red','voice_mocha','yupi_fruit','twistko','voice_choco','voice_waffle','null']

frame_counter = 0
cached_prediction = None

def preprocess_image(image_data):
    try:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            print("Invalid or empty image")
            return None

        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize the image
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
# def preprocess_image(image_data):
#     image = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (224, 224))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = image / 255.0  # Normalize the image
#     return np.expand_dims(image, axis=0)

def predict_image(image_data):
    preprocessed_image = preprocess_image(image_data)
    prediction = model.predict(preprocessed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

@socketio.on('image')
def handle_image(data):
    global frame_counter, cached_prediction

    frame_counter += 1
    print(f"Received image - Frame count: {frame_counter}")
    if frame_counter >= 10:
        image_data = data['image']
        cached_prediction = predict_image(image_data)
        emit('prediction', {'text': cached_prediction})
        print(f"Prediction sent - Predicted text: {cached_prediction}")
        frame_counter = 0
    else:
        if cached_prediction is not None:
            emit('prediction', {'text': cached_prediction})
            print(f"Using cached prediction - Predicted text: {cached_prediction}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80, debug=True)
