import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def identify_face(image_input):
    try:
        # Load the pre-trained model
        print("Loading model...")
        model = tf.keras.models.load_model('static/face_detection_model.keras')
        print("Model loaded successfully.")
        
        # Check if the input is a file path or an image array
        if isinstance(image_input, str):
            print("Loading image from file path...")
            # Load the image from file path
            img = image.load_img(image_input, target_size=(224, 224))
            facearray = image.img_to_array(img)
            facearray = np.expand_dims(facearray, axis=0)
        elif isinstance(image_input, np.ndarray):
            print("Using image array input...")
            facearray = np.array(image_input)
            if facearray.ndim == 3:
                facearray = np.expand_dims(facearray, axis=0)
        else:
            return "Invalid image input. Provide a file path or an image array."
        
        # Normalize and preprocess the image
        facearray = facearray.astype('float32') / 255.0
        facearray = tf.image.resize_with_pad(facearray, 224, 224)
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(facearray)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        
        # Load class names
        print("Loading class names...")
        class_names = np.load('static/face_detection_model/class_names.npy')
        
        return class_names[predicted_class_index]
    
    except Exception as e:
        return f"Error during face identification: {str(e)}"

# Example usage
result = identify_face('static/faces/Sreeram_5/Sreeram_0_aug_0.jpg')
print(f"Result: {result}")
