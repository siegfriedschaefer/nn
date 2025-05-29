
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np


# --- 1. Prepare the Data ---
# Our tiny 3x3 images: 0 for black, 1 for white.
# X_train: input images
# y_train: labels (1 for vertical line, 0 for not)

# Image 1: Vertical line in the center
img_vertical_line = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
], dtype=np.float32)



def main():

    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)

    print(f"Image shape: {img_vertical_line.shape}")

    X_train = np.array([img_vertical_line], dtype=np.float32)  # Shape: (1, 3, 3, 1)
    y_train = np.array([1], dtype=np.float32)  # Label: 1 for vertical line
    X_train = X_train.reshape((1, 3, 3, 1))  # Reshape to (batch_size, height, width, channels)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print("\nSample Image (Vertical Line):\n", X_train[0, :, :, 0])
    print("\nSample Image (Vertical Line):\n", X_train)

    # --- 2. Build the Model ---
    model = keras.Sequential(
        [
            # Input shape: 3x3 image with 1 channel
            layers.InputLayer(shape=(3, 3, 1)),

            # Convolutional Layer
            # - 1 filter: We're trying to learn one simple pattern.
            # - kernel_size=(2, 2): The filter will be 2x2.
            # - activation='relu': Rectified Linear Unit activation function.
            # - padding='valid': No padding, so output size will be smaller.
            #   (3x3 input, 2x2 filter -> (3-2+1)x(3-2+1) = 2x2 output feature map)
            layers.Conv2D(
                filters=1,
                kernel_size=(2, 2),
                activation='relu',
                padding='valid',
                name='convolutional_layer'
            ),
            # Flatten Layer
            # Converts the 2D feature map from Conv2D into a 1D vector.
            # (e.g., a 2x2 feature map becomes a 1x4 vector)
            layers.Flatten(name='flatten_layer'),

            # Dense Layer (Fully Connected Layer)
            # - 1 unit: For binary classification (vertical line or not).
            # - activation='sigmoid': Sigmoid activation outputs a value between 0 and 1,
            #   which can be interpreted as a probability.
            layers.Dense(1, activation='sigmoid', name='output_layer')

        ])
    
    # Print a summary of the model architecture
    print("\n--- Model Summary ---")
    model.summary()

    plot_file_name = 'simple_cnn_model.png'
    plot_model(model,
               to_file=plot_file_name,
               show_shapes=True,        # Display shape information
               show_dtype=False,        # Optionally display data types
               show_layer_names=True,   # Display layer names
               show_layer_activations=True, # Display activation functions
               rankdir='TB',            # 'TB' for top-to-bottom, 'LR' for left-to-right
               dpi=96                   # Dots per inch
              )

    # --- 3. Compile the Model ---

if __name__ == "__main__":
    main()



