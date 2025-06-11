#!/usr/bin/env python3
"""
Test script to verify Whisper functionality
"""

import whisper
import tempfile
import soundfile as sf
import numpy as np
import requests
import time

def test_whisper_basic():
    """Test basic Whisper functionality"""
    print("Testing Whisper basic functionality...")
    
    try:
        # Clear CUDA cache before testing
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Try smaller models first
        models_to_try = ["tiny", "base", "small", "turbo"]
        model = None
        
        for model_name in models_to_try:
            try:
                print(f"Loading Whisper {model_name} model...")
                model = whisper.load_model(model_name)
                print(f"✓ Whisper {model_name} model loaded successfully")
                break
            except Exception as model_error:
                print(f"✗ Failed to load {model_name} model: {model_error}")
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        if model is None:
            print("✗ Failed to load any Whisper model")
            return False
        
        # Create a simple test audio file (silence for now)
        print("Creating test audio...")
        sample_rate = 16000
        duration = 2  # 2 seconds
        audio_data = np.zeros(sample_rate * duration, dtype=np.float32)
        
        # Add some simple tone (440 Hz - A note)
        t = np.linspace(0, duration, sample_rate * duration, False)
        audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            temp_path = temp_file.name
        
        print(f"✓ Test audio created: {temp_path}")
        
        # Test transcription
        print("Testing transcription...")
        result = model.transcribe(temp_path)
        
        print(f"✓ Transcription completed")
        print(f"  Text: '{result['text']}'")
        print(f"  Language: {result.get('language', 'unknown')}")
        print(f"  Duration: {result.get('duration', 0):.2f}s")
        
        # Clean up
        import os
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        # Clean up CUDA memory
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

def test_stt_endpoint():
    """Test the STT endpoint"""
    print("\nTesting STT endpoint...")
    
    try:
        # Create a simple test audio file with actual speech-like pattern
        print("Creating test speech audio...")
        sample_rate = 16000
        duration = 1  # 1 second
        
        # Create a more complex waveform that might be recognized as speech
        t = np.linspace(0, duration, sample_rate * duration, False)
        
        # Combine multiple frequencies to simulate speech
        audio_data = (
            0.1 * np.sin(2 * np.pi * 200 * t) +  # Base frequency
            0.05 * np.sin(2 * np.pi * 400 * t) +  # Harmonic
            0.03 * np.sin(2 * np.pi * 800 * t) +  # Higher harmonic
            0.01 * np.random.normal(0, 1, len(t))  # Add some noise
        )
        
        # Apply envelope to make it more speech-like
        envelope = np.exp(-3 * t) * (1 - np.exp(-10 * t))
        audio_data *= envelope
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            temp_path = temp_file.name
        
        print(f"✓ Test audio created: {temp_path}")
        
        # Test the endpoint
        print("Calling STT endpoint...")
        
        with open(temp_path, 'rb') as audio_file:
            files = {'audio_file': ('test.wav', audio_file, 'audio/wav')}
            response = requests.post('http://localhost:5000/api/stt', files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ STT endpoint successful")
            print(f"  Text: '{result.get('text', '')}'")
            print(f"  Language: {result.get('language', 'unknown')}")
            print(f"  Duration: {result.get('duration', 0):.2f}s")
            print(f"  Success: {result.get('success', False)}")
            return True
        else:
            print(f"✗ STT endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing STT endpoint: {e}")
        return False

def main():
    """Main test function"""
    print("=== Whisper STT Testing ===\n")
    
    # Test basic Whisper functionality
    basic_test = test_whisper_basic()
    
    if basic_test:
        # Test the API endpoint
        time.sleep(1)  # Brief pause
        endpoint_test = test_stt_endpoint()
        
        if basic_test and endpoint_test:
            print("\n✓ All tests passed! Whisper STT is working correctly.")
        else:
            print("\n✗ Some tests failed. Check the logs above.")
    else:
        print("\n✗ Basic Whisper test failed. STT may not be available.")

if __name__ == "__main__":
    main() 