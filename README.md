# LightChat

A real-time chat application built with FastAPI and modern web technologies.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### 🛠️ Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lightchat
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the development server**
   ```bash
   python -m backend
   ```

   The API will be available at `http://127.0.0.1:8000`

## 📚 API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Alternative documentation: `http://127.0.0.1:8000/redoc`

## 🏗️ Project Structure

```
lightchat/
├── backend/               # Backend source code
│   └── main.py            # Main FastAPI application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 📝 License

This project is licensed under the MIT License.
