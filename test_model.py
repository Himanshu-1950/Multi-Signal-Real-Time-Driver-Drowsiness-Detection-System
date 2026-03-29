# Save this as test_model.py in your Colab1 folder and run it
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

model_eye = load_model('cnnEye.keras')

# Test with a black image (simulates closed eye)
black = np.zeros((24,24,1), dtype=np.float32)
pred = model_eye.predict(black.reshape(1,24,24,1), verbose=0)[0]
print(f"Black image (closed eye sim): {pred}")
print(f"Index 0: {pred[0]*100:.1f}%  Index 1: {pred[1]*100:.1f}%")
print(f"Predicted class: {np.argmax(pred)}")

# Test with a white image (simulates open eye)
white = np.ones((24,24,1), dtype=np.float32)
pred2 = model_eye.predict(white.reshape(1,24,24,1), verbose=0)[0]
print(f"\nWhite image (open eye sim): {pred2}")
print(f"Index 0: {pred2[0]*100:.1f}%  Index 1: {pred2[1]*100:.1f}%")
print(f"Predicted class: {np.argmax(pred2)}")