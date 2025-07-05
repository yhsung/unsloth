# Apple Silicon Support for Unsloth

This document describes Apple Silicon (M1, M2, M3, M4) support in Unsloth.

## 🍎 What's Supported

### ✅ Working Features
- **Apple Silicon Detection**: Automatic detection of M-series chips
- **MPS Backend**: Uses Metal Performance Shaders for GPU acceleration
- **Full Precision Training**: 16-bit and 32-bit model training
- **bfloat16 Support**: Native bfloat16 support on Apple Silicon Neural Engine
- **Memory Optimization**: Apple Silicon-specific memory management
- **All Model Types**: Llama, Mistral, Qwen, Gemma, and other supported models

### ⚠️ Limitations
- **4-bit Quantization**: Not yet supported (bitsandbytes limitation)
- **8-bit Quantization**: Not yet supported (bitsandbytes limitation)
- **xformers**: Limited support on Apple Silicon

## 📦 Installation

### Standard Installation
```bash
pip install unsloth[apple-silicon]
```

### Manual Installation
```bash
pip install unsloth
# Optionally install MLX for additional Apple Silicon optimizations
pip install mlx>=0.20.0
```

## 🚀 Usage

```python
import torch
from unsloth import FastLanguageModel

# Unsloth will automatically detect Apple Silicon and use MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b",
    max_seq_length = 2048,
    dtype = torch.bfloat16,  # Use bfloat16 for best performance on Apple Silicon
    load_in_4bit = False,    # 4-bit not supported yet on Apple Silicon
    device_map = device,
)

# Enable LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    max_seq_length = 2048,
)
```

## 🔧 Apple Silicon Optimizations

### Memory Management
- Unified memory architecture support
- Optimized memory allocation patterns
- Reduced memory fragmentation

### Performance Tips
1. **Use bfloat16**: Best performance on Apple Silicon Neural Engine
2. **Avoid Quantization**: Use full precision models for now
3. **Batch Size**: Start with smaller batch sizes and scale up
4. **Sequence Length**: Apple Silicon handles longer sequences well

### Example Training Configuration
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = SFTConfig(
        per_device_train_batch_size = 2,  # Start small on Apple Silicon
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = False,           # Use bfloat16 instead
        bf16 = True,            # Better for Apple Silicon
        logging_steps = 1,
        optim = "adamw_8bit",   # 8-bit optimizer for memory efficiency
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        dataloader_num_workers = 0,  # Disable multiprocessing on Apple Silicon
    ),
)
```

## 🧪 Testing

Run Apple Silicon compatibility tests:
```bash
python tests/test_apple_silicon.py
```

## 🔮 Future Roadmap

### Short Term
- [ ] MLX integration for additional optimizations
- [ ] Custom quantization implementation for Apple Silicon
- [ ] Memory usage profiling and optimization

### Long Term
- [ ] Native Metal kernels for key operations
- [ ] Apple Silicon specific model architectures
- [ ] Integration with Core ML for deployment

## 🐛 Known Issues

1. **xformers compatibility**: Some xformers operations may fall back to slower implementations
2. **Memory monitoring**: MPS memory usage monitoring is limited compared to CUDA
3. **Multi-GPU**: Apple Silicon doesn't support multi-GPU training

## 💡 Tips for Best Performance

1. **Close other apps**: Free up memory for training
2. **Monitor temperature**: Apple Silicon throttles under heavy load
3. **Use Activity Monitor**: Check memory pressure and GPU usage
4. **Batch size**: Find the sweet spot for your specific model and Mac

## 📞 Support

If you encounter issues with Apple Silicon support:

1. Check that MPS is available: `torch.backends.mps.is_available()`
2. Verify PyTorch version: requires PyTorch 2.0+
3. Monitor memory usage during training
4. Consider using smaller models or batch sizes

For additional help, please open an issue on the [Unsloth GitHub repository](https://github.com/unslothai/unsloth/issues) with the "Apple Silicon" label.