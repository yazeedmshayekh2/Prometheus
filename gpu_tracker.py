#!/usr/bin/env python3
"""
Comprehensive GPU Usage Tracker for Meta-Llama-3.1-8B-Instruct-AWQ-INT4 Model
This script provides detailed GPU monitoring and model testing capabilities.
"""

import time
import psutil
import torch
import threading
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import subprocess
import json

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install pynvml")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

class GPUTracker:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_data = []
        
        # Initialize NVML if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
                self.device_count = pynvml.nvmlDeviceGetCount()
                print(f"NVML initialized successfully. Found {self.device_count} GPU(s)")
            except Exception as e:
                print(f"Failed to initialize NVML: {e}")
                self.nvml_available = False
        else:
            self.nvml_available = False
        
        # Check PyTorch CUDA
        self.torch_cuda_available = torch.cuda.is_available()
        if self.torch_cuda_available:
            print(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"PyTorch version: {torch.__version__}")
        else:
            print("PyTorch CUDA not available")

    def get_gpu_info_nvidia_smi(self) -> Dict[str, Any]:
        """Get GPU info using nvidia-smi command"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 9:
                            gpu_info.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_total_mb': int(parts[2]),
                                'memory_used_mb': int(parts[3]),
                                'memory_free_mb': int(parts[4]),
                                'gpu_utilization': int(parts[5]) if parts[5] != '[Not Supported]' else 0,
                                'temperature': int(parts[6]) if parts[6] != '[Not Supported]' else 0,
                                'power_draw': float(parts[7]) if parts[7] != '[Not Supported]' else 0,
                                'power_limit': float(parts[8]) if parts[8] != '[Not Supported]' else 0
                            })
                return {'gpus': gpu_info, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            print(f"Error getting GPU info via nvidia-smi: {e}")
        return {'gpus': [], 'timestamp': datetime.now().isoformat()}

    def get_gpu_info_pytorch(self) -> Dict[str, Any]:
        """Get GPU info using PyTorch"""
        if not self.torch_cuda_available:
            return {'error': 'CUDA not available in PyTorch'}
        
        try:
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory
                
                gpu_info.append({
                    'index': i,
                    'name': props.name,
                    'memory_total_bytes': memory_total,
                    'memory_allocated_bytes': memory_allocated,
                    'memory_reserved_bytes': memory_reserved,
                    'memory_free_bytes': memory_total - memory_reserved,
                    'memory_total_mb': memory_total // (1024**2),
                    'memory_allocated_mb': memory_allocated // (1024**2),
                    'memory_reserved_mb': memory_reserved // (1024**2),
                    'memory_free_mb': (memory_total - memory_reserved) // (1024**2),
                    'major': props.major,
                    'minor': props.minor,
                    'multi_processor_count': props.multi_processor_count
                })
            return {'gpus': gpu_info, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            return {'error': f'PyTorch GPU info error: {e}'}

    def get_gpu_info_nvml(self) -> Dict[str, Any]:
        """Get GPU info using NVML"""
        if not self.nvml_available:
            return {'error': 'NVML not available'}
        
        try:
            gpu_info = []
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0
                
                # GPU utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    memory_util = util.memory
                except:
                    gpu_util = 0
                    memory_util = 0
                
                # Power
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power_draw = 0
                
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                except:
                    power_limit = 0
                
                # Name
                try:
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                except:
                    name = f"GPU {i}"
                
                gpu_info.append({
                    'index': i,
                    'name': name,
                    'memory_total_bytes': mem_info.total,
                    'memory_used_bytes': mem_info.used,
                    'memory_free_bytes': mem_info.free,
                    'memory_total_mb': mem_info.total // (1024**2),
                    'memory_used_mb': mem_info.used // (1024**2),
                    'memory_free_mb': mem_info.free // (1024**2),
                    'temperature': temp,
                    'gpu_utilization': gpu_util,
                    'memory_utilization': memory_util,
                    'power_draw_watts': power_draw,
                    'power_limit_watts': power_limit
                })
            return {'gpus': gpu_info, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            return {'error': f'NVML error: {e}'}

    def print_gpu_status(self):
        """Print comprehensive GPU status"""
        print("\n" + "="*80)
        print(f"GPU STATUS AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # NVIDIA-SMI info
        smi_info = self.get_gpu_info_nvidia_smi()
        if smi_info.get('gpus'):
            print("\n📊 NVIDIA-SMI Information:")
            for gpu in smi_info['gpus']:
                print(f"  GPU {gpu['index']}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB ({gpu['memory_used_mb']/gpu['memory_total_mb']*100:.1f}%)")
                print(f"    Utilization: {gpu['gpu_utilization']}%")
                print(f"    Temperature: {gpu['temperature']}°C")
                print(f"    Power: {gpu['power_draw']:.1f}W / {gpu['power_limit']:.1f}W")
        
        # PyTorch info
        pytorch_info = self.get_gpu_info_pytorch()
        if not pytorch_info.get('error'):
            print("\n🔥 PyTorch CUDA Information:")
            for gpu in pytorch_info['gpus']:
                print(f"  GPU {gpu['index']}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_reserved_mb']}/{gpu['memory_total_mb']} MB reserved ({gpu['memory_reserved_mb']/gpu['memory_total_mb']*100:.1f}%)")
                print(f"    Allocated: {gpu['memory_allocated_mb']} MB")
                print(f"    Compute Capability: {gpu['major']}.{gpu['minor']}")
        else:
            print(f"\n🔥 PyTorch CUDA: {pytorch_info['error']}")
        
        # NVML info
        nvml_info = self.get_gpu_info_nvml()
        if not nvml_info.get('error'):
            print("\n⚡ NVML Detailed Information:")
            for gpu in nvml_info['gpus']:
                print(f"  GPU {gpu['index']}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB ({gpu['memory_used_mb']/gpu['memory_total_mb']*100:.1f}%)")
                print(f"    Utilization: GPU {gpu['gpu_utilization']}%, Memory {gpu['memory_utilization']}%")
                print(f"    Temperature: {gpu['temperature']}°C")
                print(f"    Power: {gpu['power_draw_watts']:.1f}W / {gpu['power_limit_watts']:.1f}W")
        else:
            print(f"\n⚡ NVML: {nvml_info['error']}")

    def start_monitoring(self, interval: float = 2.0):
        """Start continuous GPU monitoring"""
        if self.monitoring:
            print("Monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Started GPU monitoring (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop continuous GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("Stopped GPU monitoring")

    def _monitor_loop(self, interval: float):
        """Internal monitoring loop"""
        while self.monitoring:
            try:
                self.print_gpu_status()
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(interval)

    def test_llama_model(self, model_name: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"):
        """Test loading and running the Llama model"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers library not available. Install with: pip install transformers")
            return False
        
        print(f"\n🤖 Testing Llama Model: {model_name}")
        print("="*60)
        
        # Clear cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print initial GPU status
        print("📊 Initial GPU Status:")
        self.print_gpu_status()
        
        try:
            print(f"\n⏳ Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✅ Tokenizer loaded successfully")
            
            print(f"\n⏳ Loading model...")
            # Print GPU status before model loading
            print("📊 GPU Status before model loading:")
            pytorch_info = self.get_gpu_info_pytorch()
            if not pytorch_info.get('error'):
                for gpu in pytorch_info['gpus']:
                    print(f"  GPU {gpu['index']}: {gpu['memory_reserved_mb']}/{gpu['memory_total_mb']} MB ({gpu['memory_reserved_mb']/gpu['memory_total_mb']*100:.1f}%)")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ Model loaded successfully")
            
            # Print GPU status after model loading
            print("\n📊 GPU Status after model loading:")
            self.print_gpu_status()
            
            # Test generation
            print(f"\n🔄 Testing text generation...")
            prompt = "Explain what GPU memory management is important for large language models."
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            print("📊 GPU Status during generation:")
            pytorch_info = self.get_gpu_info_pytorch()
            if not pytorch_info.get('error'):
                for gpu in pytorch_info['gpus']:
                    print(f"  GPU {gpu['index']}: {gpu['memory_reserved_mb']}/{gpu['memory_total_mb']} MB ({gpu['memory_reserved_mb']/gpu['memory_total_mb']*100:.1f}%)")
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print("✅ Generation completed!")
            print(f"\n📝 Response:\n{response}")
            
            # Final GPU status
            print("\n📊 Final GPU Status:")
            self.print_gpu_status()
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # Print GPU status on error
            print("\n📊 GPU Status on error:")
            self.print_gpu_status()
            
            return False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nReceived interrupt signal. Cleaning up...')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    tracker = GPUTracker()
    
    print("GPU Tracker for Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    print("="*50)
    print("Available commands:")
    print("  monitor - Start real-time GPU monitoring")
    print("  test    - Test Llama model loading and generation")
    print("  stress  - Run stress test")
    print("  help    - Show this help message")
    print("="*50)
    
    # Show initial status
    tracker.print_gpu_status()
    
    while True:
        try:
            cmd = input("\nEnter command (status/monitor/test/quit): ").strip().lower()
            
            if cmd == 'quit' or cmd == 'q':
                break
            elif cmd == 'status' or cmd == 's':
                tracker.print_gpu_status()
            elif cmd == 'monitor' or cmd == 'm':
                print("Starting continuous monitoring. Press Ctrl+C to stop.")
                try:
                    tracker.start_monitoring(interval=2.0)
                    # Keep main thread alive
                    while tracker.monitoring:
                        time.sleep(1)
                except KeyboardInterrupt:
                    tracker.stop_monitoring()
                    print("\nMonitoring stopped.")
            elif cmd == 'test' or cmd == 't':
                success = tracker.test_llama_model()
                if success:
                    print("✅ Model test completed successfully!")
                else:
                    print("❌ Model test failed!")
            elif cmd == 'stress' or cmd == 'st':
                # Implement stress test
                print("Stress test not implemented yet.")
            elif cmd == 'help' or cmd == 'h':
                print("Available commands:")
                print("  monitor - Start real-time GPU monitoring")
                print("  test    - Test Llama model loading and generation")
                print("  stress  - Run stress test")
                print("  help    - Show this help message")
            else:
                print("Unknown command. Use: status, monitor, test, stress, or help")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break
    
    tracker.stop_monitoring()
    print("Goodbye!")

if __name__ == "__main__":
    main() 