# 🛡️ Scam Detection ML API

A production-ready machine learning API for detecting scam messages across multiple modalities (SMS, WhatsApp, calls, and audio transcripts).

## 🚀 Features

- **Multi-modal Detection**: Supports SMS, WhatsApp, call transcripts, and audio transcripts
- **DistilBERT Model**: Fast and accurate transformer-based classification
- **Production Ready**: Includes CORS, error handling, health checks, and logging
- **Risk Levels**: Categorizes threats as LOW, MEDIUM, or HIGH
- **RESTful API**: Built with FastAPI for high performance

## 📋 Prerequisites

- Python 3.8+
- pip

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd nlp_junior
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- FastAPI and Uvicorn (API framework)
- PyTorch (CPU version for inference)
- Transformers and Accelerate (Hugging Face libraries)
- Pandas and scikit-learn (data processing)
- Pydantic (data validation)

## 🎯 Training the Model

Before running the API, you need to train the model:

1. **Preprocess the data**
```bash
cd training
python preprocess.py
```

This will:
- Load data from all sources (SMS, WhatsApp, calls, audio)
- Clean and normalize text
- Create `processed_data.csv`

2. **Train the model**
```bash
python train.py
```

This will:
- Split data into train/validation sets (80/20)
- Fine-tune DistilBERT on your data
- Save the model to `../model/scam_model/`
- Show training progress and validation metrics

Training typically takes 10-30 minutes depending on your hardware.

## 🏃 Running the API

1. **Start the server**
```bash
    cd backend
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

2. **Access the API**
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## 📡 API Endpoints

### Health Check
```bash
GET /
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "scam-detection-api",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Predict Scam
```bash
POST /predict
```

Request body:
```json
{
  "text": "Congratulations! You've won $1000. Click here to claim your prize!",
  "modality": "sms"
}
```

Response:
```json
{
  "is_scam": 1,
  "confidence": 0.923,
  "risk_level": "HIGH",
  "recommended_action": "DO_NOT_INTERACT",
  "modality": "sms"
}
```

**Modality options**: `sms`, `whatsapp`, `call`, `audio`

## 🧪 Testing the API

### Using curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your account has been suspended. Click here to verify.",
    "modality": "sms"
  }'
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "Your package is waiting. Pay $5 shipping fee.",
        "modality": "whatsapp"
    }
)

print(response.json())
```

## 🚢 Deployment to Railway

1. **Ensure all files are committed**
```bash
git add .
git commit -m "Ready for deployment"
git push
```

2. **Deploy to Railway**
- Connect your GitHub repository to Railway
- Railway will automatically detect `railway.json`
- The app will build and deploy automatically

3. **Environment Variables** (if needed)
- No environment variables required for basic setup
- For production, consider adding API keys for authentication

## 🎯 Deployment to Render

For detailed Render deployment instructions, see **[RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md)**

### Quick Start:

1. **Train the model locally first** (Render free tier has limited resources):
   ```bash
   cd training
   python preprocess.py
   python train.py
   ```

2. **Commit the trained model**:
   ```bash
   git add -f model/scam_model/
   git commit -m "Add trained model"
   git push
   ```

3. **Deploy to Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Click **New +** → **Web Service**
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml` and deploy

4. **Test your deployment**:
   ```bash
   curl https://your-app.onrender.com/health
   ```

**Important:** See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for complete instructions, troubleshooting, and handling large model files.

## 📊 Model Performance

The model is evaluated on a validation set during training:
- **Architecture**: DistilBERT (66M parameters)
- **Training**: 3 epochs with 80/20 train/val split
- **Metrics**: Logged during training (check console output)

## 🛠️ Project Structure

```
nlp_junior/
├── backend/
│   └── app.py              # FastAPI application
├── data/                   # Raw CSV datasets
│   ├── public_sms.csv
│   ├── public_whatsapp.csv
│   ├── public_calls.csv
│   └── public_audio_transcripts.csv
├── training/
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Model training
│   └── processed_data.csv  # Generated after preprocessing
├── model/                  # Generated after training
│   └── scam_model/         # Trained model files
├── requirements.txt        # Python dependencies
├── railway.json           # Railway deployment config
└── README.md              # This file
```

## 🔒 Security Notes

- **CORS**: Currently set to allow all origins (`*`). Configure appropriately for production.
- **Rate Limiting**: Consider adding rate limiting for production use.
- **Authentication**: Add API key authentication for production deployments.

## 🐛 Troubleshooting

### Model not loading
```
Error: Model directory not found
```
**Solution**: Train the model first using `cd training && python train.py`

### Import errors
```
ModuleNotFoundError: No module named 'pydantic'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### CORS errors
**Solution**: Update `allow_origins` in `backend/app.py` to include your frontend domain

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 📧 Contact

[Add contact information here]
