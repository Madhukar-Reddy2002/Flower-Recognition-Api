import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image

# Define model path and class names (adjust as needed)
MODEL_PATH = "./MyModel.h5"
CLASS_NAMES = [
    "balloon flower", "black-eyed susan", "foxglove", "frangipani", "jasmine",
    "lotus lotus", "orange hibiscus", "orange marigold", "oxeye daisy",
    "pink hibiscus", "pink rose", "red hibiscus", "redRose", "stemless gentian",
    "sunflower", "thorn apple", "water lily", "yellow hibiscus", "yellow marigold",
    "yellow rose"
]

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model(MODEL_PATH)

# Define a Pydantic model for the API request body
class PredictionRequest(BaseModel):
    image: UploadFile  # Field to receive the uploaded image

# Create a FastAPI instance with CORS configuration for wide accessibility
app = FastAPI(
    docs_url="/docs",
    redoc_url=None,  # Disable ReDoc for cleaner documentation
    default_response_model=object,  # Allow for flexible response structures
)

# Function to preprocess the image for prediction
def preprocess_image(image_file: UploadFile):
    try:
        image = Image.open(image_file.file)
        image = image.resize((250, 250))  # Adjust image size as needed by your model
        image_array = np.array(image)
        image_batch = np.expand_dims(image_array, axis=0)
        return image_batch
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

# API endpoint to handle image classification requests with CORS support
@app.post("/classify", tags=["Prediction"])
async def classify(file: UploadFile = File(...)):
    try:
        # Process uploaded image
        image_batch = preprocess_image(file)

        # Make prediction
        prediction = model.predict(image_batch)
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]  # Get indices of top 3 classes sorted by confidence
        top_3_classes = [CLASS_NAMES[i] for i in top_3_indices]
        top_3_confidences = [float(prediction[0][i]) for i in top_3_indices]

        # Return prediction response with top 3 classes and confidence
        return {
            "predicted_class": top_3_classes[0],  # Most likely class
            "confidence": top_3_confidences[0],
            "top_3_predictions": [
                {"class_name": c, "confidence": f"{conf:.2%}"}
                for c, conf in zip(top_3_classes, top_3_confidences)
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Entry route for API documentation and basic information
@app.get("/", tags=["API Info"])
async def root():
    return {
        "message": "Welcome to the Flower Classification API! Use the `/classify` endpoint to upload an image and get its predicted flower class with confidence scores.",
        "endpoints": {
            "/classify": {
                "method": "POST",
                "description": "Classify a flower image",
                "request": {"body": {"image": "Image file (JPEG or PNG)"}},
                "response": {
                    "200": {
                        "description": "Successful classification",
                        "example": {
                            "predicted_class": "sunflower",
                            "confidence": 0.98,
                            "top_3_predictions": [
                                {"class_name": "sunflower", "confidence": "98.00%"},
                                {"class_name": "orange hibiscus", "confidence": "1.50%"},
                                {"class_name": "red hibiscus", "confidence": "0.50%"},
                            ],
                        },
                    },
                    "400": {"description": "Invalid image format or missing image file"},
                },
            },
        },
    }

# Run the FastAPI server using uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
