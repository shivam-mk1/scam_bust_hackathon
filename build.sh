#!/bin/bash

# Render build script
echo "🔧 Starting build process..."

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Train the model (if not already present)
echo "🤖 Checking for model..."
if [ ! -d "model/scam_model" ]; then
    echo "⚠️  Model not found. Training model..."
    echo "📊 Preprocessing data..."
    cd training
    python preprocess.py
    
    echo "🎯 Training model (this may take 10-20 minutes)..."
    python train.py
    
    cd ..
    echo "✅ Model training complete!"
else
    echo "✅ Model already exists, skipping training"
fi

echo "✅ Build complete!"
    