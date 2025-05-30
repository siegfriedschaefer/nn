
import numpy as np


img_vertical_line = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
], dtype=np.float32)



def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


class Conv2DLayer:
    def __init__(self, num_filters, kernel_size, input_channels=1, stride=1, padding=0):
        self.num_filters = num_filters # For this example, we'll use 1
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.stride = stride
        self.padding = padding # Assuming padding=0 (valid convolution) for simplicity

        rng = np.random.default_rng()

        # Initialize kernel(s) with small random numbers
        # Shape: (num_filters, input_channels, kernel_height, kernel_width)
        # For our simple case: (1, 1, kernel_size, kernel_size)

        self.kernels = rng.standard_normal(size=(num_filters, input_channels, kernel_size, kernel_size)) * 0.1

        # future test - for a single 2x2 filter, no need for random initialization
        # self.kernels = np.zeros((1,1,2,2), dtype=np.int8)  
        # No bias for this simple conv layer

        # Variables to store for backpropagation
        self.last_input = None
        self.last_pre_activation_output = None # Output before ReLU

        print(f"Initialized Conv2DLayer with {num_filters} filters, kernel size {kernel_size}, input channels {input_channels}, stride {stride}, padding {padding}")
        print("Kernels.dtype:", self.kernels.dtype)
        print("Initial kernels shape:", self.kernels.shape)
        print("Initial kernels values:\n", self.kernels)

    def forward_pass(self, input_data):
        # input_data shape: (batch_size, channels, height, width)
        # For single image: (1, channels, height, width)
        # Our example will pass one image at a time, so effectively (channels, height, width)
        self.last_input = input_data

        print("Input data shape:", input_data.shape)
        print("Input data ndim:", input_data.ndim)

        # Ensure input_data has batch dimension
        if input_data.ndim == 3: # (channels, height, width)
            input_data = input_data[np.newaxis, ...] # Add batch dim: (1, channels, height, width)
        
        n_batch, in_channels, in_height, in_width = input_data.shape
        k_height, k_width = self.kernel_size, self.kernel_size

        out_height = (in_height - k_height) // self.stride + 1
        out_width = (in_width - k_width) // self.stride + 1

        # Output before activation
        self.last_pre_activation_output = np.zeros((n_batch, self.num_filters, out_height, out_width))

        for b in range(n_batch): # Should be 1 for our single image pass
            for f in range(self.num_filters):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + k_height
                        w_start = w_out * self.stride
                        w_end = w_start + k_width

                        # Extract the receptive field from input
                        receptive_field = input_data[b, :, h_start:h_end, w_start:w_end]
                        
                        # Perform convolution (element-wise product and sum)
                        # Kernel shape is (input_channels, k_height, k_width) for a single filter
                        self.last_pre_activation_output[b, f, h_out, w_out] = \
                            np.sum(receptive_field * self.kernels[f, :, :, :])
        
        # Apply ReLU activation, 
        # for now just dummy implementation 
        activated_output = relu(self.last_pre_activation_output)
        return activated_output.squeeze(axis=0) # Remove batch dim if it was 1

class FlattenLayer:
    def __init__(self):
        self.last_input_shape = None

    def forward_pass(self, input_data):
        # input_data shape: (channels, height, width) or (batch, channels, height, width)
        self.last_input_shape = input_data.shape
        if input_data.ndim > 1:
             return input_data.reshape(input_data.shape[0], -1) if input_data.ndim > 3 else input_data.flatten()
        return input_data.flatten()


X_train_list = [
    img_vertical_line[np.newaxis, :, :],  
]

# Conv Layer: 1 filter, 2x2 kernel
conv_layer = Conv2DLayer(num_filters=1, kernel_size=2, input_channels=1)
conv_layer.forward_pass(X_train_list[0])



