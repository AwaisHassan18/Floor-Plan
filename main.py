from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
import os
import json
import random
from paddleocr import PaddleOCR
import langid
from dotenv import load_dotenv
import pymongo

load_dotenv()

mongo_uri =os.getenv("MONGO_URI")
client = pymongo.MongoClient(mongo_uri)
db = client["floorplan"]

language_collection = db["language"]
user_details = db["User Sing up"]
label_model = db["Labels model"]
nlabel_model = db["No labels Model"]




load_dotenv(override=True)
API_KEYS = [
    "yml7axkU1o1fk4Y6zDGZ",
    "ttMXN8V7PUZZiCyTuAfn",
    "wH9uxWraipjfSTTFr6pd",
    "X11chiCHualmkXnycNwi",
    "IUg1rN4zXRmiPp367KsF",
    "4UfP9BfXydgz2ZkGYg8p",
    "zKddXvFU23vVtNXzupk7",
    "lQ71C8cYDKwPKXfEQFTN",
    "45tTp4LaVhIgTtFldnH2",
    "ueypNqSFkSMiqeexnxhX"
]
ROBOFLOW_API_KEY = random.choice(API_KEYS)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rf = Roboflow(api_key=ROBOFLOW_API_KEY)





class JSONEncoder(json.JSONEncoder):
    """Extend json-encoder class"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return json.JSONEncoder.default(self, obj)



# Without-label Model
@app.post("/no-label-model")
async def processed_image(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    dimension = (677, 577)
    resized_image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

    project = rf.workspace().project("roomdetector")
    model = project.version(5).model
    result = model.predict(resized_image, confidence=25, overlap=30).json()
    project1 = rf.workspace().project("builderformer")
    model1 = project1.version(9).model
    result1 = model1.predict(resized_image, confidence=40, overlap=30).json()
    project2 = rf.workspace().project("capstone-7fuz5")
    model2 = project2.version(4).model
    result2 = model2.predict(resized_image, confidence=25, overlap=30).json()
    project3 = rf.workspace().project("delp")
    model3 = project3.version(1).model
    result3 = model3.predict(resized_image, confidence=5, overlap=30).json()

    filteredData = dataFilteration(result, ['room'])
    filteredData1 = dataFilteration(result1, ['door', 'window'])
    filteredData2 = dataFilteration(result2, ['Walls'])
    filteredData3 = dataFilteration(result3, ['BT Entry Point', 'Cat 6 Data Socket',
                                              'Ceiling Mounted Continuous Extract Fan With Boost Mode Activated By Light Switch',
                                              'Ceiling Mounted Continuous Extract Fan With Local Boost Switch',
                                              'Co-Ax TV Socket', 'Consumer Unit', 'Double Socket', 'Electric Meter Box',
                                              'External Wall Light', 'Full Height Tiling', 'Fused spur',
                                              'Gas Meter Box', 'Grid Switch', 'Hob Switch', 'Internal Wall Light',
                                              'Light Switch', 'Low Energy Downlighter', 'Low Energy Pendant Light',
                                              'Mains Wired Smoke Detector', 'Outside Socket', 'Outside Tap',
                                              'Oven Switch', 'Programmable Room Thermostat', 'Radiator',
                                              'Recirculating Extractor Fan', 'Shaver Socket', 'Single Socket',
                                              'TV - Satellite Multisocket', 'Telephone Socket', 'Track Light',
                                              'Twin LED Strip Light', 'USB Double Socket',
                                              'Underfloor Heating Manifold', 'Water Entry Position'])

    # Concatenate all the lists into one
    all_filtered_data = filteredData + filteredData1 + filteredData2 + filteredData3

    # Insert all the filtered data into the MongoDB collection
    nlabel_model.insert_many(all_filtered_data)

    response_data = json.loads(json.dumps(all_filtered_data, cls=JSONEncoder))

    # concatenated_data = []
    # concatenated_data.extend(filteredData)
    # concatenated_data.extend(filteredData1)
    # concatenated_data.extend(filteredData2)
    # concatenated_data.extend(filteredData3)
    #
    # return concatenated_data
    return JSONResponse(content=response_data)


# language Detection
def extract_text_from_image(image_path):
    ocr_model = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=True)
    img = cv2.imread(image_path)
    result = ocr_model.ocr(img)

    # Log the result to understand its structure
    print("OCR Result:", result)

    if result:
        try:
            # Attempt to process the OCR result assuming it's in the expected format
            extracted_text = '\n'.join([line[1][0] for line in result[0]])
            return extracted_text, image_path
        except Exception as e:
            # Handle cases where the result might not be in the expected format
            return f"Error processing the OCR results: {str(e)}", image_path
    else:
        return "No text detected in the image.", image_path




def detect_language(text):
    # Detecting the language of the text
    lang, confidence = langid.classify(text)
    return lang, confidence


def store_text_and_language(image_path, text, language, confidence):
    document = {
        "image_path": image_path,
        "extracted_text": text,
        "detected_language": language,
        "confidence": confidence
    }
    language_collection.insert_one(document)


@app.post("/extract-text/")
async def extract_text(image_file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_path = "temp_image.png"
    with open(temp_path, "wb") as buffer:
        buffer.write(await image_file.read())
    # Extract text from the image
    extracted_text, image_path = extract_text_from_image(temp_path)
    if extracted_text not in ["No text detected in the image.", "Error in text extraction."]:
        language, confidence = detect_language(extracted_text)
        store_text_and_language(image_path, extracted_text, language, confidence)

        # Process further, e.g., detect language, etc.
        return {"extracted_text": extracted_text, "detected_language": language}
    else:
        return {"error": extracted_text}


def dataFilteration(result, list):
    filtered_data = []
    for item in result["predictions"]:
        if item["class"] in list:
            filtered_data.append({
                "x": item["x"],
                "y": item["y"],
                "height": item["height"],
                "width": item["width"],
                "class": obfuscatedClass(item["class"])
            })
    return filtered_data


def obfuscatedClass(item):
    if item == 'Light Switch':
        item = 'Switched Outlet'
    elif item == 'Co-Ax TV Socket' or item == 'TV - Satellite Multisocket':
        item = 'TV Outlet'
    elif item == 'Electric Meter Box' or item == 'Gas Meter Box':
        item = 'Meter Box'
    elif item == 'Light Switch' or item == 'Grid Switch':
        item = 'Switch (single)'
    elif item == 'BT Entry Point' or item == 'Telephone Socket':
        item = 'Phone Jack'
    elif item == 'Outside Socket' or item == 'Single Socket' or item == 'Double Socket' or item == 'Shaver Socket':
        item = 'Receptacle 220v Outlet'
    elif item == 'Programmable Room Thermostat':
        item = 'Thermostat'
    elif item == 'Internal Wall Light' or item == 'External Wall Light':
        item = 'Wall Light/Sconce'
    elif item == 'Ceiling Mounted Continuous Extract Fan With Boost Mode Activated By Light Switch' or item == 'Ceiling Mounted Continuous Extract Fan With Local Boost Switch':
        item = 'Ceiling Fan'
    elif item == 'Hob Switch' or item == 'Oven Switch':
        item = 'Range'
    elif item == 'Cat 6 Data Socket' or item == 'USB Double Socket':
        item = 'Floor Box with Power & Data'
    return item


# With-label Model
@app.post("/all-in-one/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    dimension = (677, 577)
    resized_image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

    project = rf.workspace().project("builderformer")
    model = project.version(9).model
    result = model.predict(resized_image, confidence=40, overlap=30).json()
    project1 = rf.workspace().project("capstone-7fuz5")
    model1 = project1.version(4).model
    result1 = model1.predict(resized_image, confidence=30, overlap=30).json()
    project2 = rf.workspace().project("room-detection-a5anb")
    model2 = project2.version(2).model
    result2 = model2.predict(resized_image, confidence=15, overlap=30).json()
    project3 = rf.workspace().project("delp")
    model3 = project3.version(1).model
    result3 = model3.predict(resized_image, confidence=5, overlap=30).json()
    print("hello saadi")
    filteredData = dataFilteration(result, ['door', 'window'])
    filteredData1 = dataFilteration(result1, ['Walls'])
    filteredData2 = dataFilteration(result2,
                                    ['Living Room', 'Bedroom', 'kitchen', 'garage', 'Bathroom', 'Balcony', 'staircase',
                                     'Wardrobe', 'DiningRoom', 'Staircase', 'Master Bedroom', 'office', 'StoreRoom',
                                     'Utility', 'PrayerRoom', 'BalconyRailing', 'StudyRoom', 'Office'])
    filteredData3 = dataFilteration(result3, ['BT Entry Point', 'Cat 6 Data Socket',
                                              'Ceiling Mounted Continuous Extract Fan With Boost Mode Activated By '
                                              'Light Switch',
                                              'Ceiling Mounted Continuous Extract Fan With Local Boost Switch',
                                              'Co-Ax TV Socket', 'Consumer Unit', 'Double Socket', 'Electric Meter Box',
                                              'External Wall Light', 'Full Height Tiling', 'Fused spur',
                                              'Gas Meter Box', 'Grid Switch', 'Hob Switch', 'Internal Wall Light',
                                              'Light Switch', 'Low Energy Downlighter', 'Low Energy Pendant Light',
                                              'Mains Wired Smoke Detector', 'Outside Socket', 'Outside Tap',
                                              'Oven Switch', 'Programmable Room Thermostat', 'Radiator',
                                              'Recirculating Extractor Fan', 'Shaver Socket', 'Single Socket',
                                              'TV - Satellite Multisocket', 'Telephone Socket', 'Track Light',
                                              'Twin LED Strip Light', 'USB Double Socket',
                                              'Underfloor Heating Manifold', 'Water Entry Position'])
    print("filteration error")
    all_filtered_data = filteredData + filteredData1 + filteredData2 + filteredData3

    # Insert all the filtered data into the MongoDB collection
    label_model.insert_many(all_filtered_data)

    response_data = json.loads(json.dumps(all_filtered_data, cls=JSONEncoder))

    # concatenated_data = []
    # concatenated_data.extend(filteredData3)
    # concatenated_data.extend(filteredData2)
    # concatenated_data.extend(filteredData1)
    # concatenated_data.extend(filteredData)
    #
    # return concatenated_data
    return JSONResponse(content=response_data)




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=200)
