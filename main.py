#!/usr/bin/env python3
"""
Digital Inspector - AI Document Detection System (Main Entry Point)
Armeta CV Hackathon - Complete Solution

This file orchestrates the complete workflow:
1. Dataset preparation (PDF + JSON → YOLO format)
2. Model training (YOLOv8s with optimized settings for small objects)
3. Hybrid inference (YOLO + OpenCV QR detection)
4. Testing and evaluation
5. Web application launch
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json

def run_command(cmd, description="Running command", capture_output=True):
    """Execute a shell command with optional output capture"""
    print(f"🤖 {description}")
    print(f"   Command: {cmd}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ❌ Error: {result.stderr}")
                return False, result.stderr
            else:
                print("   ✅ Success")
                return True, result.stdout
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0, ""
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return False, str(e)

def prepare_dataset(args):
    """Prepare dataset from PDFs and annotations"""
    print("\n" + "="*70)
    print("📄 STEP 1: Preparing Dataset")
    print("="*70)
    
    if args.kaggle:
        # Kaggle specific paths
        cmd = "python prepare_dataset.py"
        success, output = run_command(cmd, "Preparing dataset for Kaggle")
    else:
        # Local paths - need to make sure they exist
        data_dir = Path("data")
        if not data_dir.exists():
            print("❌ 'data' directory doesn't exist. Please create it with 'pdf' and 'annotations' subdirectories.")
            return False
        
        cmd = "python prepare_dataset.py"
        success, output = run_command(cmd, "Preparing dataset locally")
    
    if success:
        print("   Dataset prepared successfully!")
        print("   └── Output: /kaggle/working/data/ (Kaggle) or ./data/ (local)")
        return True
    else:
        print("   ❌ Dataset preparation failed")
        return False

def train_model(args):
    """Train the YOLO model"""
    print("\n" + "="*70)
    print("🏋️  STEP 2: Training Model")
    print("="*70)
    
    cmd = "python train_yolov8s_optimized.py"
    success, output = run_command(cmd, "Training YOLOv8s model")
    
    if success:
        print("   Model training completed!")
        print("   └── Best model: runs/detect/train/weights/best.pt")
        return True
    else:
        print("   ❌ Model training failed")
        return False

def run_inference(args):
    """Run inference with the trained model"""
    print("\n" + "="*70)
    print("🔍 STEP 3: Running Inference")
    print("="*70)
    
    # Basic command - can be extended with source argument
    if args.source:
        cmd = f"python inference_optimized.py --source {args.source} --model {args.model} --output {args.output}"
        if args.no_opencv:
            cmd += " --no-opencv"
    else:
        # Run with default settings
        cmd = f"python inference_optimized.py --source data/test/ --model {args.model} --output {args.output}"
        if args.no_opencv:
            cmd += " --no-opencv"
    
    success, output = run_command(cmd, "Running hybrid inference (YOLO + OpenCV)")
    
    if success:
        print(f"   Inference completed! Results saved to: {args.output}")
        return True
    else:
        print("   ❌ Inference failed")
        return False

def run_tests(args):
    """Run various tests"""
    print("\n" + "="*70)
    print("🧪 STEP 4: Running Tests")
    print("="*70)
    
    test_scripts = []
    if args.test_type == "all":
        test_scripts = ["test_model.py", "test_model_local.py", "test_model_overlapping.py"]
    elif args.test_type == "local":
        test_scripts = ["test_model_local.py"]
    elif args.test_type == "overlapping":
        test_scripts = ["test_model_overlapping.py"]
    else:
        test_scripts = ["test_model_local.py"]  # default
    
    all_passed = True
    for test_script in test_scripts:
        cmd = f"python {test_script}"
        success, output = run_command(cmd, f"Running {test_script}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("   All tests passed!")
        return True
    else:
        print("   Some tests failed")
        return False

def launch_web_app(args):
    """Launch the Streamlit web application"""
    print("\n" + "="*70)
    print("🌐 STEP 5: Launching Web Application")
    print("="*70)
    
    print("🚀 Starting Streamlit app...")
    print("   Open your browser at: http://localhost:8501")
    print("   Press Ctrl+C to stop the server")
    
    try:
        # Run streamlit in the foreground so user can see the output
        cmd = "streamlit run streamlit_app.py"
        os.system(cmd)
        return True
    except KeyboardInterrupt:
        print("\n   ✋ Web app stopped by user")
        return True

def show_project_info():
    """Display project information and metrics"""
    print("\n" + "="*70)
    print("📊 DIGITAL INSPECTOR - PROJECT INFORMATION")
    print("="*70)
    
    try:
        with open('final_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"Project: {results.get('project', 'N/A')}")
        print(f"Hackathon: {results.get('hackathon', 'N/A')}")
        print(f"Model: {results.get('model', 'N/A')}")
        
        print(f"\n🎯 Key Metrics:")
        metrics = results.get('metrics_validation', {}).get('overall', {})
        print(f"   - mAP50: {metrics.get('mAP50', 'N/A')}")
        print(f"   - mAP50-95: {metrics.get('mAP50-95', 'N/A')}")
        print(f"   - Precision: {metrics.get('precision', 'N/A')}")
        print(f"   - Recall: {metrics.get('recall', 'N/A')}")
        
        print(f"\n📈 Per-Class Performance:")
        per_class = results.get('metrics_validation', {}).get('per_class', {})
        for class_name, metrics in per_class.items():
            if 'mAP50' in metrics:
                print(f"   - {class_name}: mAP50 = {metrics['mAP50']}")
        
    except FileNotFoundError:
        print("   final_results.json not found - run training to see metrics")
    except json.JSONDecodeError:
        print("   Error reading final_results.json")

def main():
    parser = argparse.ArgumentParser(description='Digital Inspector - Complete AI Document Detection System')
    parser.add_argument('--mode', type=str, choices=['prepare', 'train', 'inference', 'test', 'app', 'all', 'info'], 
                       default='info', help='Operation mode')
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle-specific paths')
    parser.add_argument('--source', type=str, default='data/test/', help='Source for inference (image, folder, or PDF)')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help='Model path for inference')
    parser.add_argument('--output', type=str, default='results.json', help='Output file for inference results')
    parser.add_argument('--no-opencv', action='store_true', help='Disable OpenCV QR detection (YOLO only)')
    parser.add_argument('--test-type', type=str, choices=['all', 'local', 'overlapping'], 
                       default='local', help='Type of tests to run')
    
    args = parser.parse_args()
    
    print("🔍 DIGITAL INSPECTOR - AI Document Detection System")
    print("   Armeta CV Hackathon 2024")
    print("   Hybrid: YOLOv8s + OpenCV QRCodeDetector")
    
    if args.mode == 'info':
        show_project_info()
        return
    
    # Validate dependencies
    try:
        import ultralytics
        import cv2
        import numpy as np
        import torch
        print("✅ Dependencies validated")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Run: pip install -r requirements.txt")
        return
    
    # Execute workflow based on mode
    if args.mode == 'prepare':
        prepare_dataset(args)
    elif args.mode == 'train':
        if prepare_dataset(args):  # Prepare first if not already done
            train_model(args)
    elif args.mode == 'inference':
        run_inference(args)
    elif args.mode == 'test':
        run_tests(args)
    elif args.mode == 'app':
        launch_web_app(args)
    elif args.mode == 'all':
        # Complete workflow: prepare -> train -> inference -> test -> app
        if prepare_dataset(args):
            if train_model(args):
                if run_inference(args):
                    run_tests(args)
                    print("\n" + "="*70)
                    print("🎉 COMPLETE WORKFLOW FINISHED!")
                    print("   All steps completed successfully")
                    print("="*70)
                    
                    # Optionally launch web app after everything
                    response = input("   Launch web app? (y/n): ")
                    if response.lower() in ['y', 'yes']:
                        launch_web_app(args)
                else:
                    print("   ❌ Inference step failed, stopping workflow")
            else:
                print("   ❌ Training step failed, stopping workflow")
        else:
            print("   ❌ Data preparation failed, stopping workflow")
    
    print("\n✨ Digital Inspector workflow completed!")

if __name__ == '__main__':
    main()