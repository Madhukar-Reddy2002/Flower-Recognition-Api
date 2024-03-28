# Flower Classification API with FastAPI

## Overview

This README file provides an overview of a FastAPI project designed to classify flower images using a pre-trained TensorFlow model.

### Features

- Classifies flower images into 20 distinct flower categories.
- Leverages FastAPI for a clean and efficient API structure.
- Supports Cross-Origin Resource Sharing (CORS) for requests from various origins.
- Includes error handling mechanisms with appropriate HTTP error codes.
- Offers OpenAPI documentation for straightforward interaction with the API.

### Requirements

- Python 3.x [Download Python](https://www.python.org/downloads/windows/)
- FastAPI [FastAPI GitHub](https://github.com/tiangolo/fastapi)
- Pydantic [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- TensorFlow [TensorFlow Installation](https://www.tensorflow.org/install/pip)
- Pillow [Pillow PyPI](https://pypi.org/project/pillow/)
- NumPy [NumPy Website](https://numpy.org/)

## Installation

1. Create a virtual environment to isolate dependencies (recommended).
2. Install the required libraries:

    ```bash
    pip install fastapi pydantic tensorflow pillow numpy
    ```

## Model Preparation

- Obtain or train your pre-trained TensorFlow model (saved as MyModel.h5 in this example).
- Ensure the model architecture and input/output formats align with the code's assumptions.

## Usage

1. Start the API server:

    ```bash
    uvicorn main:app --reload  # Adjust the filename if necessary
    ```

2. Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
3. Use a tool like Postman or curl to test the API:

    **cURL Example:**

    ```bash
    curl -X POST http://127.0.0.1:8000/classify -F "file=@your_image.jpg"
    ```

    Replace `your_image.jpg` with the path to your image file. The response will be a JSON object containing the predicted class name and confidence score.

## Code Breakdown

1. **Imports**: Necessary libraries for building the API, model loading, and image processing.

2. **Application Setup**: Creates a FastAPI instance (app) and configures CORS middleware to allow cross-origin requests.

3. **Model Loading**: Loads the pre-trained TensorFlow model from MyModel.h5 and defines a list of flower class names (CLASS_NAMES) corresponding to the model's output.

4. **Prediction Response Model**: Defines a Pydantic model (PredictionResponse) to structure the API response with class_info (predicted class name) and confidence (prediction confidence score).

5. **Welcome Route (/)**: Redirects root route (/) to the API documentation for ease of access.

6. **Classification Route (/classify)**:
    - Accepts Image: Expects a file upload named file.
    - Preprocessing:
        - Reads the image file contents.
        - Opens the image using Pillow.
        - Resizes the image to the model's expected input size (250x250 pixels).
        - Normalizes pixel values between 0 and 1.
        - Expands the dimension of the image array for compatibility with the model.
    - Prediction:
        - Makes a prediction using the loaded model.
        - Identifies the class with the highest probability.
    - Response:
        - Creates a dictionary containing the predicted class name and confidence score.
        - Returns a JSON response with the prediction results using the PredictionResponse model.
    - Error Handling:
        - Catches potential exceptions during processing.
        - Raises an HTTPException with status code 500 (Internal Server Error) and the exception message for debugging.

## Further Considerations

- Explore advanced error handling and logging for better API monitoring and debugging.
- Implement input validation to ensure users upload images in the expected format and size.
- Consider handling multiple image uploads in a single request.
- For production deployment, choose a suitable server technology like Gunicorn or uWSGI to run the FastAPI application.