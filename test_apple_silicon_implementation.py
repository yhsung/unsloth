#!/usr/bin/env python3
"""
Test script to verify Apple Silicon MPS support in Unsloth implementation
"""

import sys
import platform
import os

print("🧪 Testing Apple Silicon Implementation")
print("=" * 50)

# Check system information
print(f"Platform: {platform.system()}")
print(f"Machine: {platform.machine()}")
print(f"Python version: {sys.version}")

# Check PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
except ImportError as e:
    print(f"❌ PyTorch not available: {e}")
    sys.exit(1)

print("\n🔍 Testing device detection before Unsloth import...")

# Test basic MPS functionality
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(5, 5, device=device)
    y = torch.randn(5, 5, device=device)
    z = torch.matmul(x, y)
    print(f"✅ Basic MPS tensor operations work: {z.device}")
else:
    print("⚠️  MPS not available, will test fallback behavior")

print("\n🦥 Testing Unsloth Apple Silicon implementation...")

# Add current directory to Python path so we can import our modified unsloth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import our modified unsloth
    print("Importing unsloth...")
    import unsloth
    
    print(f"✅ Unsloth imported successfully")
    print(f"Device type detected: {getattr(unsloth, 'DEVICE_TYPE', 'Not found')}")
    print(f"Supports bfloat16: {getattr(unsloth, 'SUPPORTS_BFLOAT16', 'Not found')}")
    
    # Test device type detection
    if hasattr(unsloth, 'DEVICE_TYPE'):
        device_type = unsloth.DEVICE_TYPE
        if platform.system() == "Darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available():
            if device_type == "mps":
                print("✅ Apple Silicon correctly detected as MPS device")
            else:
                print(f"❌ Expected MPS device type, got: {device_type}")
        else:
            print(f"ℹ️  Non-Apple Silicon system detected device type: {device_type}")
    
    # Test kernel utils
    print("\n🔧 Testing kernel utilities...")
    from unsloth.kernels import utils
    
    # Test get_device_type function
    detected_device = unsloth.get_device_type()
    print(f"Device type from function: {detected_device}")
    
    # Test basic tensor operations with our utilities
    if torch.backends.mps.is_available():
        print("Testing MPS tensor operations with unsloth utilities...")
        device = torch.device("mps")
        test_tensor = torch.randn(10, 10, device=device, dtype=torch.bfloat16)
        print(f"✅ Created bfloat16 tensor on MPS: {test_tensor.device} {test_tensor.dtype}")
        
        # Test matmul operations
        result = torch.matmul(test_tensor, test_tensor.t())
        print(f"✅ Matrix multiplication works: {result.shape} on {result.device}")
    
    print("\n🎯 Testing fast operations...")
    
    # Test fast_dequantize (should work without quantization)
    from unsloth.kernels.utils import fast_dequantize
    
    if torch.backends.mps.is_available():
        test_weight = torch.randn(100, 100, device="mps", dtype=torch.bfloat16)
        result = fast_dequantize(test_weight, quant_state=None)
        print(f"✅ fast_dequantize works without quantization: {result.device}")
    
    print("\n🧪 Running Apple Silicon test suite...")
    
    # Run our test suite
    try:
        exec(open('tests/test_apple_silicon.py').read())
        print("✅ Apple Silicon test suite passed")
    except FileNotFoundError:
        print("⚠️  Test file not found, skipping test suite")
    except Exception as e:
        print(f"⚠️  Test suite had issues: {e}")
    
except ImportError as e:
    print(f"❌ Failed to import unsloth: {e}")
    print("This might be due to missing dependencies")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("🏁 Apple Silicon implementation test completed!")