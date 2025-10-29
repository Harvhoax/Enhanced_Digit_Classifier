import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import io

# Page configuration
st.set_page_config(
    page_title="Extended Digit Recognition (0-99)",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('extended_digit_recognition_model.h5')
        return model, 100  # 100 classes (0-99)
    except:
        try:
            # Fallback to original single digit model
            model = keras.models.load_model('digit_recognition_model.h5')
            return model, 10  # 10 classes (0-9)
        except:
            st.error("Model file not found! Please ensure model file is in the same directory.")
            return None, 0

model, num_classes = load_model()
is_extended = (num_classes == 100)  

# Title and description
st.title("🔢 Extended Handwritten Digit Recognition System (0-99)")
st.markdown(f"""
This application uses a **Convolutional Neural Network (CNN)** trained to recognize 
handwritten numbers from **0 to {num_classes-1}** with high accuracy.
""")

if is_extended:
    st.info("✨ **Extended Model Active**: This model can recognize numbers from 0 to 99!")
else:
    st.warning("⚠️ Using basic model (0-9 only). Train the extended model for 0-99 recognition.")

# Sidebar
with st.sidebar:
    st.header("📊 About")
    if is_extended:
        st.info("""
        **Extended Model Architecture:**
        - 4 Convolutional Blocks
        - Batch Normalization
        - Dropout Regularization
        - Dense Layers: 512, 256, 128 units
        - Input: 28×56 pixels (double width)
        
        **Recognition Range:** 0-99
        
        **Expected Accuracy:** ~97%+
        """)
    else:
        st.info("""
        **Model Architecture:**
        - 3 Convolutional Blocks
        - Batch Normalization
        - Dropout Regularization
        - Dense Layers with 256 & 128 units
        
        **Dataset:** MNIST (60,000 training images)
        
        **Expected Accuracy:** ~99%+
        """)
    
    st.header("🎨 How to Use")
    st.markdown("""
    1. **Draw Mode:** Draw digits in the canvas
       - Single digit: Draw centered
       - Two digits: Draw side by side
    2. **Upload Mode:** Upload an image
    3. Click **Predict** to see the result
    4. View confidence scores and top predictions
    """)
    
    if is_extended:
        st.header("💡 Tips")
        st.markdown("""
        **For best results:**
        - Draw single digits centered
        - For two digits, draw left digit first, then right
        - Keep digits separated but not too far apart
        - Use clear, bold strokes
        """)

# Main content
tab1, tab2, tab3 = st.tabs(["✏️ Draw Digit", "📤 Upload Image", "📈 Model Info"])

# Tab 1: Draw Digit
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Draw a number (0-{num_classes-1})")
        
        # Adjust canvas width for extended model
        canvas_width = 560 if is_extended else 280
        
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=18 if is_extended else 20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=canvas_width,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_button = st.button("🎯 Predict Number", key="predict_draw")
        with col_btn2:
            clear_button = st.button("🗑️ Clear Canvas", key="clear_canvas")
    
    with col2:
        st.subheader("Prediction Results")
        result_placeholder = st.empty()
        chart_placeholder = st.empty()
        details_placeholder = st.empty()
    
    if predict_button and canvas_result.image_data is not None:
        # Process the drawn image
        img = canvas_result.image_data.astype('uint8')
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        # Resize based on model type
        if is_extended:
            img_resized = cv2.resize(img_gray, (56, 28), interpolation=cv2.INTER_AREA)
        else:
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize
        img_normalized = img_resized.astype('float32') / 255.0
        
        if is_extended:
            img_input = img_normalized.reshape(1, 28, 56, 1)
        else:
            img_input = img_normalized.reshape(1, 28, 28, 1)
        
        # Check if image is not empty
        if np.sum(img_resized) > 0:
            if model is not None:
                # Make prediction
                prediction = model.predict(img_input, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                # Get top 5 predictions
                top_5_idx = np.argsort(prediction[0])[-5:][::-1]
                top_5_probs = prediction[0][top_5_idx] * 100
                
                # Display results
                with result_placeholder.container():
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h1 style='text-align: center; color: #4CAF50; font-size: 72px;'>{predicted_class:02d if is_extended else predicted_class}</h1>
                        <h3 style='text-align: center;'>Confidence: {confidence:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Plot top 5 predictions
                with chart_placeholder:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#4CAF50' if i == 0 else '#87CEEB' for i in range(5)]
                    bars = ax.barh(range(5), top_5_probs, color=colors)
                    ax.set_yticks(range(5))
                    ax.set_yticklabels([f'{idx:02d}' if is_extended else str(idx) for idx in top_5_idx])
                    ax.set_xlabel('Probability (%)', fontsize=11)
                    ax.set_title('Top 5 Predictions', fontsize=13, fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                # Show detailed probabilities
                with details_placeholder:
                    st.subheader("Top 5 Predictions:")
                    for i, (idx, prob) in enumerate(zip(top_5_idx, top_5_probs)):
                        label = f"{idx:02d}" if is_extended else str(idx)
                        st.write(f"**{i+1}.** Number **{label}**: {prob:.2f}%")
        else:
            st.warning("⚠️ Please draw a number first!")

# Tab 2: Upload Image
with tab2:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Upload an image of a number (0-{num_classes-1})")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Predict button
            if st.button("🎯 Predict Number", key="predict_upload"):
                # Preprocess image
                img_gray = ImageOps.grayscale(image)
                
                # Resize based on model type
                if is_extended:
                    img_resized = img_gray.resize((56, 28))
                else:
                    img