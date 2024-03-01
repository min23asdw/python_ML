from flask import Flask
from flask_socketio import SocketIO, emit, disconnect
import base64
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app)

# Load your Keras model
model = tf.keras.models.load_model('../model/model_forAPP.h5', compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["categorical_accuracy"],
)

# Class labels for your model
class_labels = ['dewberry_blue', 'creco','Ahh','rollercoaster_cheese','bengbeng','bento','deno_stone','chocopie','rollercoaster_spicy','kitkat','lays_3','lays_cheese','lays_original','dewberry_red','lay_green','ff','oreo','pringles_green','lotus','marujo_red','marujo_green','tilli_indigo','tasto_spicy','tasto_honey','snackjack_chicken','snackjack_saltpepper','tawan_Larb','snackjack_shell','tilli_blue','tilli_red','voice_mocha','yupi_fruit','twistko','voice_choco','voice_waffle','null']

# Global variables for prediction
frame_counter = 0
cached_prediction = None


def preprocess_image(image_data):
    try:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = image / 255.0  # Normalize the image
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def predict_image(image_data):
    preprocessed_image = preprocess_image(image_data)
    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image)
        top_classes = np.argsort(predictions)[0, ::-1][:5]
        confidence_scores = predictions[0, top_classes]
        confidence_scores_percent = confidence_scores * 100 / np.sum(confidence_scores)
        textdis = {}
        for i in range(5):
            textdis[i] = f"{class_labels[top_classes[i]]}:{confidence_scores_percent[i]:.2f}%"
        predict_text = "#".join(textdis.values())
        return predict_text
    else:
        return None


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('image')
def handle_image(data):
    global frame_counter, cached_prediction

    image_data = data['image']
    # print(data)
    # print(image_data)
    print(f"Received image - Frame count: {frame_counter}")
    frame_counter += 1

    if frame_counter % 2 == 0:  # Only predict every 10 frames
        cached_prediction = predict_image(image_data)
        if cached_prediction:
            socketio.start_background_task(send_prediction, cached_prediction)
            print(f"Prediction sent - Predicted text: {cached_prediction}")
        else:
            print("Prediction not available")
    else:
        # Send cached prediction back to Flutter
        if cached_prediction:
            socketio.start_background_task(send_prediction, cached_prediction)
            print(f"Sending cached prediction - Predicted text: {cached_prediction}")
        else:
            print("Cached prediction not available")


def send_prediction(prediction):
    try:
        socketio.emit('prediction', {'text': prediction})
    except AssertionError as e:
        # Handle write() before start_response error
        print(f"Error sending prediction: {e}")
        print("Ignoring write() before start_response")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80, debug=True)
