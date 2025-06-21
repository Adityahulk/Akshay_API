import os
import uuid
import shutil
import pathlib

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyAZunR_fXNJQZSXaN07o--W9v1eu6cDV34")
app = FastAPI()

EXTRACTION_PROMPT_TEMPLATE = """
You are a document analysis expert.

Extract structured data from the following {file_type} content and return it in {output_format} format.

⚠️ Extraction Rules:
- Understand context even if layout is complex (tables, headings, nested info).
- For CSV: include column headers and rows in proper table structure.
- For JSON: use key-value pairs with nested structure if needed.
- DO NOT add extra notes, explanations, or markdown – just return raw {output_format} content.

Start extracting:
"""


@app.post("/process-file/")
async def process_file(
    file: UploadFile = File(...),
    output_format: str = Form("json")  # text | json | csv
):
    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_bytes = pathlib.Path(temp_path).read_bytes()

        # Determine file type and MIME
        if file.filename.lower().endswith(".pdf"):
            mime_type = "application/pdf"
            file_type = "PDF"
        elif file.filename.lower().endswith(".csv"):
            mime_type = "text/csv"
            file_type = "CSV"
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type. Please upload a PDF or CSV."})

        # Generate prompt
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(file_type=file_type, output_format=output_format)

        # Call Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                prompt
            ]
        )

        result = response.text.strip()
        ext = "txt" if output_format == "text" else output_format
        output_filename = f"output_{uuid.uuid4()}.{ext}"

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(result)

        return FileResponse(output_filename, media_type="application/octet-stream", filename=output_filename)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
