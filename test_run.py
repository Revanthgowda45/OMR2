#!/usr/bin/env python3
"""
Quick test script for OMR Hackathon application
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'opencv-python-headless', 
        'numpy',
        'pandas',
        'pillow',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python-headless':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif package == 'pillow':
                from PIL import Image
                print(f"✅ Pillow: {Image.__version__}")
            else:
                __import__(package)
                print(f"✅ {package}: installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: missing")
    
    return missing_packages

def install_packages(packages):
    """Install missing packages"""
    if not packages:
        return True
        
    print(f"\nInstalling missing packages: {', '.join(packages)}")
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("\n🚀 Starting OMR Evaluation System...")
    print("📍 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop\n")
    
    try:
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py']
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True

def main():
    """Main function"""
    print("🎯 OMR Evaluation System - Code4Edtech Hackathon")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists('streamlit_app.py'):
        print("❌ streamlit_app.py not found in current directory")
        print("📁 Make sure you're in the OMR_Hackathon folder")
        return False
    
    print("📁 Found streamlit_app.py")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n📦 Installing {len(missing)} missing packages...")
        if not install_packages(missing):
            return False
    
    print("\n✅ All dependencies ready!")
    
    # Run application
    run_streamlit()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1)
