import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        
        # Normalize image (same as in training pipeline)
        test_image = test_image / 255.0
        
        # Get prediction probabilities
        prediction_probs = model.predict(test_image)
        result = np.argmax(prediction_probs, axis=1)
        print(result)
        print(f"Prediction probabilities: {prediction_probs}")

        # Map class index to label and get confidence
        class_names = ['Coccidiosis', 'Healthy']
        predicted_class = class_names[result[0]]
        confidence = prediction_probs[0][result[0]]
        
        # Return structured response with confidence scores
        return [{
            "image": predicted_class,
            "prediction": {
                class_names[0]: float(prediction_probs[0][0]),
                class_names[1]: float(prediction_probs[0][1])
            }
        }]