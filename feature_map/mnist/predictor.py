# For Python 3.6 we use the base keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config.config_data import EXPECTED_LABEL
from config.config_dirs import MODEL_MNIST
from feature_map.mnist.utils.general import reshape


class Predictor:
    # Load the pre-trained model.
    model = keras.models.load_model(MODEL_MNIST)
    print("Loaded model from disk")

    @staticmethod
    def predict(img):
        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, 10)
        explabel = np.argmax(explabel.squeeze())

        # Predictions vector
        try:
            img_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(reshape(img)))
            predictions = Predictor.model.predict(img_rgb)
        except:
            # The model expects grayscale images
            predictions = Predictor.model.predict(tf.convert_to_tensor(reshape(img)))

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        if prediction1 != EXPECTED_LABEL:
            confidence_notclass = predictions[0][prediction1]
        else:
            confidence_notclass = predictions[0][prediction2]

        confidence = confidence_expclass - confidence_notclass

        return prediction1, confidence

    @staticmethod
    def prediction_probability(img):
        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, 10)
        explabel = np.argmax(explabel.squeeze())

        # Predictions vector
        try:
            img_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(reshape(img)))
            predictions = Predictor.model.predict(img_rgb)
        except:
            # The model expects grayscale images
            predictions = Predictor.model.predict(tf.convert_to_tensor(reshape(img)))

        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        return confidence_expclass
