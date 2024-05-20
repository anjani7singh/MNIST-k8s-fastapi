from fastapi import FastAPI,File, UploadFile, HTTPException
import uvicorn
from PIL import Image
from predictor import mnist_classifier
import io
import sys

app= FastAPI(title="Cars 24 Assignment API")
classifier = mnist_classifier()

@app.post('/getDigit')
async def get_digit(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(
            status_code=400, detail=f'File \'{file.filename}\' is not an image.')
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        output = classifier.predict(image)
        return {"status_code": 200,"output": output}

    except Exception as error:
        print(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def get_health():
    return {"status_code": 200,"output":"healthy"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
    