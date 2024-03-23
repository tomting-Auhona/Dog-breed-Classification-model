import tkinter as tk  # Import the tkinter library for GUI development
from tkinter import filedialog, messagebox  # Import filedialog and messagebox for user interactions
from PIL import ImageTk, Image  # Import ImageTk and Image from PIL for image handling
import numpy as np  # Import numpy for numerical operations
from keras.models import load_model  # Import load_model from keras.models to load trained models
import keras.utils as image  # Import utils from keras to preprocess images
from keras.applications.mobilenet_v2 import preprocess_input  # Import preprocess_input from MobileNetV2 for image preprocessing

# Load the trained model
model = load_model('dog_v2.h5')

# Mapping class indices to class names
class_mapping = {0: 'Afghan', 1: 'African Wild Dog', 2: 'Airedale', 3: 'American Hairless', 4: 'American Spaniel',
                 5: 'Basenji', 6: 'Basset', 7: 'Beagle', 8: 'Bearded Collie', 9: 'Bermaise', 10: 'Bichon Frise',
                 11: 'Blenheim', 12: 'Bloodhound', 13: 'Bluetick', 14: 'Border Collie', 15: 'Borzoi', 16: 'Boston Terrier',
                 17: 'Boxer', 18: 'Bull Mastiff', 19: 'Bull Terrier', 20: 'Bulldog', 21: 'Cairn', 22: 'Chihuahua',
                 23: 'Chinese Crested', 24: 'Chow', 25: 'Clumber', 26: 'Cockapoo', 27: 'Cocker', 28: 'Collie',
                 29: 'Corgi', 30: 'Coyote', 31: 'Dalmation', 32: 'Dhole', 33: 'Dingo', 34: 'Doberman', 35: 'Elk Hound',
                 36: 'French Bulldog', 37: 'German Sheperd', 38: 'Golden Retriever', 39: 'Great Dane', 40: 'Great Perenees',
                 41: 'Greyhound', 42: 'Groenendael', 43: 'Irish Spaniel', 44: 'Irish Wolfhound', 45: 'Japanese Spaniel',
                 46: 'Komondor', 47: 'Labradoodle', 48: 'Labrador', 49: 'Lhasa', 50: 'Malinois', 51: 'Maltese',
                 52: 'Mex Hairless', 53: 'Newfoundland', 54: 'Pekinese', 55: 'Pit Bull', 56: 'Pomeranian', 57: 'Poodle',
                 58: 'Pug', 59: 'Rhodesian', 60: 'Rottweiler', 61: 'Saint Bernard', 62: 'Schnauzer', 63: 'Scotch Terrier',
                 64: 'Shar_Pei', 65: 'Shiba Inu', 66: 'Shih-Tzu', 67: 'Siberian Husky', 68: 'Vizsla', 69: 'Yorkie'}

def preprocess_image(image_path):
    # Preprocess image for model input
    img = image.load_img(image_path, target_size=(224, 224))  # Load image and resize to 224x224 pixels
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape
    img_array = preprocess_input(img_array)  # Preprocess image according to MobileNetV2 requirements
    return img_array

def predict(image_path):
    # Predict dog breed from image
    img_array = preprocess_image(image_path)  # Preprocess image
    predictions = model.predict(img_array)  # Perform prediction
    predicted_class = np.argmax(predictions)  # Get index of the predicted class
    return class_mapping[predicted_class]  # Return the predicted dog breed

def upload_image():
    # Upload image for classification
    file_path = filedialog.askopenfilename()  # Open file dialog to select an image
    if file_path:
        # Predict dog breed and display result
        predicted_class = predict(file_path)  # Get predicted dog breed
        result_label.config(text=f"Predicted Breed: {predicted_class}")  # Update result label text

# Initialize tkinter window
root = tk.Tk()  # Create main window instance
root.title("Dog Breed Prediction")  # Set window title
root.geometry("500x650")  # Set window dimensions

# Background image
bg_image = Image.open("background_image.jpg")  # Load background image
bg_image = bg_image.resize((500, 650), Image.ANTIALIAS)  # Resize background image
bg_image = ImageTk.PhotoImage(bg_image)  # Convert background image to Tkinter format

# Background label
bg_label = tk.Label(root, image=bg_image)  # Create label to display background image
bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Place background label in window

# Upload Button
upload_button = tk.Button(root, text="Upload Image", command=upload_image, bg='#EFC3CA', fg='black', font=('Comic Sans MS', 14))  # Create button for uploading image
upload_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)  # Place button at specified location in window

# Prediction result label
result_label = tk.Label(root, text="", bg='#EFC3CA', fg='black', font=('Comic Sans MS', 14))  # Create label to display prediction result
result_label.place(relx=0.5, rely=0.6, anchor=tk.CENTER)  # Place label at specified location in window

root.mainloop()  # Start the main loop to run the application
