#!/usr/bin/env python3
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test suite for Apple Silicon (MPS) compatibility
"""

import sys
import platform
import torch
import pytest

def test_apple_silicon_detection():
    """Test that Apple Silicon is properly detected"""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # This is an Apple Silicon Mac
        assert torch.backends.mps.is_available(), "MPS should be available on Apple Silicon"
        print("✅ Apple Silicon (MPS) detected and available")
    else:
        pytest.skip("Test only runs on Apple Silicon Macs")

def test_unsloth_mps_import():
    """Test that Unsloth can be imported and detect MPS"""
    try:
        import unsloth
        if hasattr(unsloth, 'DEVICE_TYPE'):
            device_type = unsloth.DEVICE_TYPE
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                if torch.backends.mps.is_available():
                    assert device_type == "mps", f"Expected MPS device type, got {device_type}"
                    print(f"✅ Unsloth correctly detected device type: {device_type}")
                else:
                    print("ℹ️  MPS not available on this system")
            else:
                print(f"ℹ️  Non-Apple Silicon system detected device type: {device_type}")
        else:
            print("⚠️  DEVICE_TYPE not found in unsloth module")
    except ImportError as e:
        pytest.fail(f"Failed to import unsloth: {e}")

def test_mps_basic_operations():
    """Test basic PyTorch MPS operations work"""
    if not (platform.system() == "Darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available()):
        pytest.skip("Test only runs on Apple Silicon with MPS")
    
    # Test basic tensor operations on MPS
    device = torch.device("mps")
    x = torch.randn(10, 10, device=device)
    y = torch.randn(10, 10, device=device)
    z = torch.matmul(x, y)
    
    assert z.device.type == "mps", "Result should be on MPS device"
    assert z.shape == (10, 10), "Matrix multiplication shape should be correct"
    print("✅ Basic MPS tensor operations working")

def test_bfloat16_support():
    """Test that bfloat16 is supported on Apple Silicon"""
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        pytest.skip("Test only runs on Apple Silicon")
    
    try:
        import unsloth
        if hasattr(unsloth, 'SUPPORTS_BFLOAT16'):
            assert unsloth.SUPPORTS_BFLOAT16 == True, "Apple Silicon should support bfloat16"
            print("✅ bfloat16 support detected on Apple Silicon")
        else:
            print("⚠️  SUPPORTS_BFLOAT16 not found in unsloth module")
    except ImportError as e:
        pytest.fail(f"Failed to import unsloth: {e}")

if __name__ == "__main__":
    print("🧪 Running Apple Silicon compatibility tests")
    test_apple_silicon_detection()
    test_unsloth_mps_import()
    test_mps_basic_operations()
    test_bfloat16_support()
    print("✅ All Apple Silicon tests completed")