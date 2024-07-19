try:
    # Try to import torch to check if it's installed
    import torch
    print("PyTorch is installed.")
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch is not installed.")
except Exception as e:
    print(f"An error occurred: {e}")


#try:
#    from core.utils import PathTools
#except ImportError:
#    print("utils is not accessable.")
#except Exception as e:
#    print(f"An error occurred: {e}")


#try:
#    # Try to import torch to check if it's installed
#    import synthspline
#    print("PyTorch is installed.")
#    print(f"PyTorch version: {torch.__version__}")
#except ImportError:
#    print("PyTorch is not installed.")
#except Exception as e:
#    print(f"An error occurred: {e}")
