# SMS Transaction Classifier

A machine learning system that extracts structured transaction information from SMS messages while continuously improving through user feedback.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Development](#development)
- [API Documentation](#api-documentation)
- [Model Management](#model-management)
- [Contributing](#contributing)
- [License](#license)

## Overview

The SMS Transaction Classifier is designed to automatically extract and classify transaction details from SMS messages. It features a continuous learning system that improves over time through user feedback and pattern recognition.

### Key Features
- 🤖 Automated transaction detail extraction
- 🔄 Real-time user feedback integration
- 📈 Continuous model improvement
- ✅ Quality control and validation
- 📊 Performance monitoring
- 🔖 Version control for models

## Architecture

### High-Level System Design
```
[SMS Input] → [Preprocessing] → [Model Prediction] → [Post-processing] → [User Interface]
                                     ↑                                         ↓
                              [Model Registry] ← [Training Pipeline] ← [Feedback System]
```

### Core Components
1. **Base Model Service**
   - Token classification model (BERT/RoBERTa)
   - Field extraction logic
   - Confidence scoring

2. **Feedback Collection System**
   - User interface for corrections
   - Feedback validation
   - Storage system

3. **Continuous Learning Pipeline**
   - Feedback aggregation
   - Incremental training
   - Model evaluation
   - Deployment automation

## Getting Started

### Prerequisites
- Python 3.8+
- PostgreSQL
- Redis
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone git@github.com:maurihimanshu/sms_classifier.git
cd sms_classifier
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python scripts/init_db.py
```

### Running the Application

1. Start the application:
```bash
python src/main.py
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## Development

### Project Structure
```
sms_classifier/
├── src/
│   ├── api/            # API endpoints and routing
│   ├── models/         # ML models and predictions
│   ├── preprocessing/  # Data preprocessing
│   ├── training/       # Training pipelines
│   ├── feedback/       # Feedback collection
│   ├── persistence/    # Model versioning and storage
│   ├── monitoring/     # System monitoring
│   ├── security/       # Security implementations
│   └── utils/          # Utility functions
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── docs/
├── docker/
└── scripts/
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit
pytest tests/integration
pytest tests/performance
```

### Code Quality
```bash
# Format code
black src tests

# Lint code
flake8 src tests
```

## API Documentation

### Prediction API
```python
POST /api/v1/predict
{
    "message": "Your a/c XX1234 debited for Rs.5000",
    "metadata": {
        "source": "sms_gateway",
        "timestamp": "2024-01-30T12:34:56Z"
    }
}
```

### Feedback API
```python
POST /api/v1/feedback
{
    "prediction_id": "pred_123",
    "corrections": {
        "transaction_amount": {
            "is_correct": true,
            "corrected_value": null
        }
    }
}
```

## Model Management

### Version Control
The system maintains multiple versions of the model with comprehensive metadata:
- Model weights and configuration
- Performance metrics
- Training history
- Environment information

### Backup and Recovery
- Automated backups before updates
- Quick recovery capabilities
- Version rollback support
- Performance validation

### Monitoring
- Real-time performance tracking
- Error rate monitoring
- Resource usage tracking
- Alert system

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation as needed
- Add type hints to functions
- Use async/await for I/O operations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BERT model from Hugging Face
- FastAPI framework
- PyTorch ecosystem
- Open-source community

## Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Open a new issue if needed

## Roadmap

- [ ] Enhanced pattern recognition
- [ ] Multi-language support
- [ ] Real-time analytics dashboard
- [ ] Advanced anomaly detection
- [ ] API client libraries