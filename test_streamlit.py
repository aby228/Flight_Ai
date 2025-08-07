#!/usr/bin/env python3
"""
Test script for Flight AI Streamlit app
Run this to verify everything works before deployment
"""

import subprocess
import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from united_airlines_enhanced_model import UnitedAirlinesEnhancedPredictor
        print("✅ Flight AI model imported successfully")
    except ImportError as e:
        print(f"❌ Flight AI model import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if the model can be instantiated"""
    print("\n🔍 Testing model creation...")
    
    try:
        from united_airlines_enhanced_model import UnitedAirlinesEnhancedPredictor
        model = UnitedAirlinesEnhancedPredictor()
        print("✅ Model instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    print("\n🔍 Testing Streamlit app...")
    
    try:
        # Import the main function from streamlit_app
        import streamlit_app
        print("✅ Streamlit app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app import failed: {e}")
        return False

def check_files():
    """Check if all required files exist"""
    print("\n🔍 Checking required files...")
    
    required_files = [
        'streamlit_app.py',
        'united_airlines_enhanced_model.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def test_requirements():
    """Test if requirements.txt is valid"""
    print("\n🔍 Testing requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        # Check for essential packages
        essential_packages = ['streamlit', 'pandas', 'numpy', 'scikit-learn']
        missing_packages = []
        
        for package in essential_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements.txt: {', '.join(missing_packages)}")
            return False
        else:
            print("✅ All essential packages in requirements.txt")
            return True
            
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def main():
    """Run all tests"""
    print("🚀 Flight AI - Streamlit Deployment Test")
    print("=" * 50)
    
    tests = [
        ("File Check", check_files),
        ("Requirements", test_requirements),
        ("Import Test", test_imports),
        ("Model Test", test_model_creation),
        ("Streamlit App Test", test_streamlit_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your app is ready for deployment.")
        print("\nTo run locally:")
        print("streamlit run streamlit_app.py")
        print("\nTo deploy:")
        print("1. Push to GitHub")
        print("2. Deploy on Streamlit Cloud")
        print("3. Share your app URL!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before deployment.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
