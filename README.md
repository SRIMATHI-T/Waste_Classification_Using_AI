# Waste Classification AI

This repository contains a waste classification AI system using a **ResNet50** model augmented with a **CBAM (Convolutional Block Attention Module)**. 

## Project Structure
- `app/`: Contains the Streamlit user interface (`streamlit_app.py`).
- `models/`: Stores the trained model weights.
- `src/`: Source code for the model architecture and data processing.
  - `src/models/`: CBAM implementation and model inference definitions.
  - `src/data/`: Image preprocessing and transforms.

## Installation

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your trained model is located at `models/resnet50_cbam_exp3.pth`.

## Running the App

To run the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```
"# Waste_Classification_Using_AI" 
