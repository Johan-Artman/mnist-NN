import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore


def create_model():
    model = Sequential([
        tf.keras.Input(shape=(784,)), # type: ignore
        Dense(256, activation='relu', name="L0"),
        Dense(128, activation='relu', name="L1"),  
        Dense(64, activation='relu', name="L2"),     
        Dense(10, activation='linear', name="OL")
    ], name='mnist_model')
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # type: ignore
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # type: ignore
        metrics=['accuracy']
    )
    return model

# Load trained model
model = create_model()
try:
    model.load_weights('mnist_model.weights.h5')
    print("Successfully loaded trained model")
except (FileNotFoundError, OSError):
    print("No trained model found. Please save your model weights from the notebook first.")
    print("In your notebook, add: model.save_weights('mnist_model.weights.h5')")
    exit(1)

# Set up the drawing board dimensions
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

# main window
root = tk.Tk()
root.title("Draw a Digit")

# Canvas widget
canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
canvas.pack()

image1 = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
draw = ImageDraw.Draw(image1)


def paint(event):
    brush_size = 15
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)

    canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    draw.ellipse([x1, y1, x2, y2], fill='black')

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_WIDTH, CANVAS_HEIGHT], fill='white')
    prediction_label.config(text="Draw a digit and click Predict")


def predict_digit():
    img = image1.resize((28, 28))
    img = ImageOps.invert(img)
    img_arr = np.array(img) / 255.0
    img_flattened = img_arr.reshape(1, 784)
    
    # Make prediction with the model
    prediction = model.predict(img_flattened, verbose=0)
    probabilities = tf.nn.softmax(prediction).numpy()[0]  # type: ignore
    digit = np.argmax(probabilities)
    confidence = probabilities[digit] * 100
    
    # Update the prediction display
    prediction_text = f"Predicted Digit: {digit}\nConfidence: {confidence:.1f}%"
    prediction_label.config(text=prediction_text)
    
    print(f"Predicted digit: {digit} (confidence: {confidence:.2f}%)")
    print(f"All probabilities: {[f'{i}: {prob*100:.1f}%' for i, prob in enumerate(probabilities)]}")

# Bind mouse drag event to the paint function
canvas.bind("<B1-Motion>", paint)

# Create prediction display box
prediction_frame = tk.Frame(root, relief="sunken", borderwidth=2)
prediction_frame.pack(pady=10)

prediction_label = tk.Label(prediction_frame, text="Draw a digit and click Predict", 
                           font=("Arial", 16), width=30, height=2)
prediction_label.pack(padx=10, pady=10)

# Create buttons to clear the canvas and trigger prediction
button_frame = tk.Frame(root)
button_frame.pack()

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
clear_button.pack(side="left", padx=10, pady=10)

predict_button = tk.Button(button_frame, text="Predict", command=predict_digit)
predict_button.pack(side="left", padx=10, pady=10)

# Run the Tkinter main loop
root.mainloop()