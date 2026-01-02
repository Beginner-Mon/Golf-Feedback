# Golf Feedback System

A comprehensive golf swing analysis application with real-time feedback, featuring a React frontend and FastAPI backend powered by computer vision and pose estimation.

## Project Structure

```
golf-feedback-system/
├── frontend/
│   └── swing-better/     # React application
├── src/                  # Backend FastAPI application
├── requirements.txt      # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- Node.js 16.x or higher
- npm or yarn package manager

## Backend Setup

### 1. Create Virtual Environment

Navigate to your project root directory and create a virtual environment:

```bash
python -m venv venv
```

Or if you're using Python 3 specifically:

```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

Activate the virtual environment based on your operating system:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your command line prompt, indicating the virtual environment is active.

### 3. Install Python Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Download Required Models

Before running the application, you need to download the required model files:

**Download Link:** [Model Files on Google Drive](https://drive.google.com/drive/folders/1BKUNltJUyj6fsY3t4qcpK0GeR5Ke8EFV?usp=sharing)

**Installation Instructions:**

1. Download the `models` folder from the Google Drive link
2. Place the downloaded `models` folder in the root directory
3. Download the `golfpose-checkpoints` folder from the Google Drive link
4. Place the downloaded `golfpose-checkpoints` folder in the `src/golfpose/` directory
   - Final path should be: `src/golfpose/golfpose-checkpoints/`

Your directory structure should look like this after downloading:
```
models/   # Downloaded model files
src/
├── app/
├── golfpose/
│   └── golfpose-checkpoints/  # Downloaded checkpoint files
└── ...
```

### 5. Run the Backend

**Development Mode (with auto-reload):**
```bash
uvicorn src.app.main:app --reload
```

**Production Mode:**
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`


## Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd frontend/swing-better
```

### 2. Install Node Dependencies

```bash
npm install
```

Or if you're using yarn:

```bash
yarn install
```

### 3. Run the Frontend

**Development Mode:**
```bash
npm run dev
```

Or with yarn:
```bash
yarn dev
```

**Production Build:**
```bash
npm run build
```

Or with yarn:
```bash
yarn build
```



The frontend application will typically be available at `http://localhost:3000` (or the port specified in your configuration).

## Running the Complete Application

To run both frontend and backend simultaneously:

1. **Terminal 1 - Backend:**
   ```bash
   # From /src root
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   uvicorn app.main:app --reload
   ```

2. **Terminal 2 - Frontend:**
   ```bash
   # From project root
   cd frontend/swing-better
   npm run dev
   ```



## Research References

This project is based on the following research papers:

1. [ArXiv Paper 2508.20491v1](https://arxiv.org/pdf/2508.20491v1)
2. [ICPR 2024 GolfPose Paper](https://minghanlee.github.io/papers/ICPR_2024_GolfPose.pdf)
3. [ArXiv Paper 1903.06528](https://arxiv.org/pdf/1903.06528)

