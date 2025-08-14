from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Donut model and processor once at startup
model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
processor = DonutProcessor.from_pretrained(model_name, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

def parse_prescription_text(text: str):
    """Parse prescription text to extract medications and dosages"""
    lines = text.split("\n")
    medications = []
    dosages = []

    # Common medication patterns
    medication_patterns = [
        r'(\w+)\s+(\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|units?))',  # Medicine 25mg
        r'(\w+)\s*:\s*(\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|units?))',  # Medicine: 25mg
        r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|ml|g|mcg|units?)',  # Medicine 25 mg
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try each pattern
        for pattern in medication_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:  # Medicine and dosage
                    med_name, dosage = match
                    medications.append(med_name.capitalize())
                    dosages.append(dosage)
                elif len(match) == 3:  # Medicine, number, unit
                    med_name, number, unit = match
                    medications.append(med_name.capitalize())
                    dosages.append(f"{number} {unit}")

    return medications, dosages

@app.post("/analyze")
async def analyze_prescription(file: UploadFile = File(...)):
    file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''

    if file_extension not in ["png", "jpg", "jpeg", "txt"]:
        raise HTTPException(status_code=400, detail="File must be an image (png/jpg/jpeg) or text file (txt)")

    try:
        contents = await file.read()

        if file_extension == "txt":
            # Handle text file
            try:
                output_text = contents.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    output_text = contents.decode('latin-1')
                except UnicodeDecodeError:
                    raise HTTPException(status_code=400, detail="Unable to decode text file. Please ensure it's in UTF-8 or Latin-1 encoding.")

            # Parse medications from text
            medications, dosages = parse_prescription_text(output_text)

        else:
            # Handle image file
            try:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

            # Prepare pixel values
            pixel_values = processor(image, return_tensors="pt").pixel_values

            # Generate text from image
            generated_ids = model.generate(pixel_values, max_length=512)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Parse medications from extracted text
            medications, dosages = parse_prescription_text(output_text)

        # Create analysis results
        analysis = []
        for i, med in enumerate(medications):
            current_dosage = dosages[i] if i < len(dosages) else "Unknown"

            # Calculate recommended dosage and safety status
            try:
                # Extract numeric value from dosage
                dosage_numbers = re.findall(r'(\d+(?:\.\d+)?)', current_dosage)
                if dosage_numbers:
                    number = float(dosage_numbers[0])
                    unit = re.search(r'(mg|ml|g|mcg|units?)', current_dosage, re.IGNORECASE)
                    unit = unit.group(1) if unit else "units"

                    # Simple safety check (this would be more sophisticated in real application)
                    if unit.lower() in ['mg', 'g']:
                        safe_threshold = 500 if unit.lower() == 'mg' else 5
                    elif unit.lower() in ['ml']:
                        safe_threshold = 50
                    else:
                        safe_threshold = 100

                    status = "Safe" if number <= safe_threshold else "Review Required"
                    recommended = f"{number*0.9:.1f} {unit}" if number > safe_threshold else current_dosage
                    info = f"Current: {current_dosage}, Recommended: {recommended}"
                else:
                    status = "Review Required"
                    info = f"Current: {current_dosage}, Unable to parse dosage"
            except:
                status = "Review Required"
                info = f"Current: {current_dosage}, Unable to calculate recommendation"

            analysis.append({
                "medicine": med,
                "status": status,
                "info": info
            })

        # If no medications found, provide a default response
        if not analysis:
            file_type = "text file" if file_extension == "txt" else "image"
            analysis.append({
                "medicine": "No medications detected",
                "status": "Review Required",
                "info": f"Please ensure the {file_type} contains clear prescription information"
            })

        return {
            "filename": file.filename,
            "file_type": file_extension,
            "extracted_text": output_text,
            "analysis": analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {e}")

@app.get("/")
def home():
    return {"message": "Backend is running!"}
