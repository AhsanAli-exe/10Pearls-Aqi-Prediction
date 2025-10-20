#!/usr/bin/env python3
"""
Deployment script for AQI Prediction System
Handles local and cloud deployment
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'requests', 'pandas', 'numpy', 'scikit-learn', 
        'joblib', 'flask', 'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        return run_command(install_cmd, "Installing missing packages")
    else:
        print("✅ All dependencies are installed")
        return True

def test_system():
    """Test the complete system"""
    print("🧪 Testing system components...")
    
    # Test real-time data integration
    test_cmd = "python3 test_real_time_integration.py"
    if not run_command(test_cmd, "Testing real-time integration"):
        return False
    
    return True

def start_api():
    """Start the Flask API"""
    print("🚀 Starting Flask API...")
    api_cmd = "python3 api.py"
    return run_command(api_cmd, "Starting Flask API")

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("🚀 Starting Streamlit dashboard...")
    dashboard_cmd = "streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
    return run_command(dashboard_cmd, "Starting Streamlit dashboard")

def create_dockerfile():
    """Create Dockerfile for containerization"""
    dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data models

# Expose ports
EXPOSE 5000 8501

# Create startup script
RUN echo '#!/bin/bash\\n\
python3 api.py &\\n\
streamlit run app.py --server.port 8501 --server.address 0.0.0.0' > start.sh && \\
    chmod +x start.sh

CMD ["./start.sh"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile created")

def create_docker_compose():
    """Create docker-compose.yml for easy deployment"""
    compose_content = """
version: '3.8'

services:
  aqi-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped

  aqi-dashboard:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - aqi-api
    environment:
      - FLASK_API_URL=http://aqi-api:5000
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose_content)
    
    print("✅ docker-compose.yml created")

def deploy_local():
    """Deploy locally"""
    print("🏠 Deploying locally...")
    
    if not check_dependencies():
        return False
    
    if not test_system():
        return False
    
    print("✅ Local deployment ready!")
    print("📝 To start the system:")
    print("   1. Terminal 1: python3 api.py")
    print("   2. Terminal 2: streamlit run app.py")
    print("   3. Open browser: http://localhost:8501")
    
    return True

def deploy_docker():
    """Deploy using Docker"""
    print("🐳 Deploying with Docker...")
    
    create_dockerfile()
    create_docker_compose()
    
    if not run_command("docker-compose build", "Building Docker images"):
        return False
    
    if not run_command("docker-compose up -d", "Starting containers"):
        return False
    
    print("✅ Docker deployment completed!")
    print("📝 Access the system:")
    print("   - API: http://localhost:5000")
    print("   - Dashboard: http://localhost:8501")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Deploy AQI Prediction System')
    parser.add_argument('--mode', choices=['local', 'docker', 'test'], 
                       default='local', help='Deployment mode')
    parser.add_argument('--test-only', action='store_true', 
                       help='Only run tests, do not deploy')
    
    args = parser.parse_args()
    
    print("🌬️ AQI Prediction System Deployment")
    print("=" * 40)
    
    if args.test_only or args.mode == 'test':
        success = test_system()
    elif args.mode == 'local':
        success = deploy_local()
    elif args.mode == 'docker':
        success = deploy_docker()
    
    if success:
        print("🎉 Deployment completed successfully!")
        sys.exit(0)
    else:
        print("❌ Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()