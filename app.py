import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide"
)

MODEL_PATH = "trained_model.keras"
IMG_SIZE = (128, 128)

class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


DISEASE_INFO = {
    "Apple___Apple_scab": "Fungal disease causing dark, scabby lesions on apple leaves and fruit.",
    "Apple___Black_rot": "Fungal infection causing leaf spots, fruit rot, and branch cankers.",
    "Apple___Cedar_apple_rust": "Rust disease producing orange/yellow spots on apple leaves.",
    "Corn_(maize)___Common_rust_": "Fungal disease causing reddish-brown pustules on corn leaves.",
    "Corn_(maize)___Northern_Leaf_Blight": "Causes long, cigar-shaped gray-green lesions on corn leaves.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Creates rectangular gray/brown lesions between corn leaf veins.",
    "Potato___Early_blight": "Dark brown spots with concentric rings, usually on older leaves.",
    "Potato___Late_blight": "Serious disease causing water-soaked lesions that turn brown or black.",
    "Tomato___Early_blight": "Fungal disease causing dark concentric lesions on tomato leaves.",
    "Tomato___Late_blight": "Destructive disease causing dark, water-soaked lesions on tomato leaves.",
    "Tomato___Septoria_leaf_spot": "Small circular leaf spots with gray centers and dark borders.",
    "Tomato___Leaf_Mold": "Yellow patches on upper leaf surface with mold growth underneath.",
    "Tomato___Bacterial_spot": "Small dark water-soaked spots often surrounded by yellow halos.",
    "Tomato___Target_Spot": "Brown circular lesions with target-like rings.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Virus causing yellowing, curling leaves, and stunted growth.",
    "Tomato___Tomato_mosaic_virus": "Causes mottled green/yellow leaf patterns and distorted growth.",
}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    original_img = np.array(image)

    resized_img = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(resized_img)
    input_arr = np.array([input_arr])  # Shape: (1, 128, 128, 3)

    return image, original_img, input_arr


def predict_image(model, input_arr):
    prediction = model.predict(input_arr, verbose=0)

    result_idx = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))

    return result_idx, confidence, prediction


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_9", pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        x = img_tensor
        conv_output = None

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            x = layer(x, training=False)

            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def create_gradcam_overlay(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = heatmap_color * alpha + original_img
    overlay = np.uint8(overlay)

    return overlay


def clean_label(label):
    return label.replace("___", " - ").replace("_", " ")


st.title("🌿 Plant Disease Recognition System")
st.markdown(
    "CNN-based plant disease classification with confidence scoring and Grad-CAM explainability."
)

with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio(
        "Select Page",
        ["Home", "Disease Recognition", "About Project", "Real-World Testing Insights"]
    )

    st.divider()
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.05
    )

model = load_model()

if app_mode == "Home":
    st.header("Welcome")
    st.markdown("""
    This project detects plant diseases from leaf images using a Convolutional Neural Network.

    Unlike a basic classifier, this version also includes:

    - prediction confidence,
    - Grad-CAM explainability,
    - uncertainty warning,
    - real-world robustness analysis,
    - and disease-specific interpretation.

    The goal is not only to predict the disease class, but also to understand **where the model is looking** while making the prediction.
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    uploaded_image = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_image is not None:
        image, original_img, input_arr = preprocess_image(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width="stretch")

        result_idx, confidence, predictions = predict_image(model, input_arr)
        predicted_class = class_name[result_idx]
        readable_class = clean_label(predicted_class)

        with col2:
            st.subheader("Prediction Result")

            if confidence < confidence_threshold:
                st.warning("Prediction confidence is low. Expert review is recommended.")
            else:
                st.success("Prediction completed successfully.")

            st.metric("Predicted Class", readable_class)
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            info = DISEASE_INFO.get(predicted_class)

            if info:
                st.info(info)
            elif "healthy" in predicted_class.lower():
                st.info("The plant appears healthy based on the model prediction.")
            else:
                st.info("No detailed disease description is currently available for this class.")

        st.divider()

        st.subheader("Grad-CAM Explainability")

        heatmap = make_gradcam_heatmap(
            input_arr,
            model,
            last_conv_layer_name="conv2d_9",
            pred_index=result_idx
        )

        if heatmap is not None:
            overlay = create_gradcam_overlay(original_img, heatmap)

            col3, col4 = st.columns(2)

            with col3:
                st.image(original_img, caption="Original Image", width="stretch")

            with col4:
                st.image(
                    overlay,
                    caption="Grad-CAM Overlay: Highlighted regions influenced the prediction",
                    width="stretch"
                )

            st.markdown("""
            **Interpretation:**  
            The highlighted regions show the parts of the leaf that contributed most strongly to the CNN prediction.
            Bright red/yellow areas indicate higher model attention, while darker areas contributed less.
            """)
        else:
            st.error("Grad-CAM could not be generated for this image.")

        st.divider()

        st.subheader("Top 5 Predictions")

        top_indices = np.argsort(predictions[0])[-5:][::-1]

        for idx in top_indices:
            st.write(f"**{clean_label(class_name[idx])}** — {predictions[0][idx] * 100:.2f}%")

elif app_mode == "About Project":
    st.header("About Dataset and Model")

    st.markdown("""
    ### Dataset

    The model is trained on a plant disease dataset containing approximately **87K RGB leaf images** across **38 classes**.

    Dataset split:

    - Training images: 70,295
    - Validation images: 17,572
    - Test images: 33

    ### Model

    The system uses a CNN trained to classify plant leaf images into healthy and diseased categories.

    ### Evaluation

    The model was evaluated using:

    - validation accuracy,
    - confusion matrix,
    - precision,
    - recall,
    - F1-score,
    - Grad-CAM visualization,
    - and real-world image testing.
    """)

elif app_mode == "Real-World Testing Insights":
    st.header("Real-World Robustness Testing")

    st.markdown("""
    The model was also tested on real-world images collected outside the curated dataset.

    These images were organized into:

    - healthy leaves,
    - known disease classes,
    - unknown plant species.

    ### Key Observations

    Although the model achieved high validation accuracy, real-world testing showed several limitations:

    - some healthy leaves were misclassified as diseased,
    - some known diseases were confused with visually similar classes,
    - out-of-distribution plants were still forced into one of the 38 known classes,
    - some wrong predictions had very high confidence.

    ### Research Insight

    These results suggest that the CNN learned strong dataset-specific visual patterns but struggled with domain shift caused by natural lighting, complex backgrounds, camera variation, and unseen plant species.

    ### Future Improvements

    - Train with more real-world field images.
    - Add an unknown-class rejection mechanism.
    - Use confidence calibration.
    - Compare CNN with EfficientNet or MobileNet.
    - Extend the project to YOLOv8-based disease localization.
    """)
