from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
model_path = "./MyModel.h5"
model = tf.keras.models.load_model(model_path)

CLASS_NAMES = [
    'balloon flower', 'bishop of llandaff', 'black-eyed susan',
    'cape flower', 'foxglove', 'geranium', 'jasmine', 'lotus lotus',
    'moon orchid', 'orange hibiscus', 'orange marigold', 'oxeye daisy',
    'pink hibiscus', 'pink rose', 'red hibiscus', 'redRose', 'sunflower',
    'yellow hibiscus', 'yellow marigold', 'yellow rose'
]

class PredictionResponse(BaseModel):
    class_info: str
    confidence: float

@app.get("/")
async def welcome():
    return RedirectResponse(url="/docs")


@app.post("/classify", response_model=PredictionResponse)
async def classify(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).resize((250, 250))
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_batch = np.expand_dims(image_array, axis=0)

        # Make prediction
        prediction = model.predict(image_batch)
        top_class_index = np.argmax(prediction[0])
        top_class_name = CLASS_NAMES[top_class_index]
        top_confidence = float(prediction[0][top_class_index])

        # Return top class and confidence
        result = {"class_info": top_class_name, "confidence": top_confidence}

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
