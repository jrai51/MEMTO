import torch

if __name__ == "__main__":
    # Check if CUDA is available
    print("CUDA Available:", torch.cuda.is_available())

    # Check the number of GPUs available
    print("Device Count:", torch.cuda.device_count())

    # Print the current device being used
    if torch.cuda.is_available():
        print("Current Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

