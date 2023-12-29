from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

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
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # show_image(image)
        image = image / 255.0  # Normalize the image
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def predict_image(image_data):
    preprocessed_image = preprocess_image(image_data)
    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image)
        # predictions = predict_image(image_data)
        top_classes = np.argsort(predictions)[0, ::-1][:5]
        confidence_scores = predictions[0, top_classes]
        confidence_scores_percent = confidence_scores * 100 / np.sum(confidence_scores)
        # Display the results
        textdis = {}
        for i in range(5):
            textdis[i] = f"{class_labels[top_classes[i]]}:{confidence_scores_percent[i]:.2f}%"
        # print(" ".join(textdis))            
        predict_text = "#".join(textdis.values())
        # predicted_class = class_labels[np.argmax(prediction)]
        return predict_text
    else:
        return None


@socketio.on('image')
def handle_image_stream_event(data):
    handle_image_stream(data)
 
def handle_image_stream(data):
    global frame_counter, cached_prediction

    image_data = data['image']
    print(f"Received image - Frame count: {frame_counter}")
    frame_counter += 1

    if frame_counter % 5 == 0:  # Only predict every 10 frames
        cached_prediction = predict_image(image_data)
        socketio.emit('prediction', {'text': cached_prediction})
          
        print(f"Prediction sent - Predicted text: {cached_prediction}")
    else:
        # Send cached prediction back to Flutter
        socketio.emit('prediction', {'text': cached_prediction})
        print(f"Sending cached prediction - Predicted text: {cached_prediction}")

def show_image(image):
    img = Image.fromarray(image)
    img.show()




if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80, debug=True)
