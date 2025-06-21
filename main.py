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

Extract structured data from the following PDF content and return it in {output_format} format.

⚠️ Extraction Rules:
- Understand context even if layout is complex (tables, headings, nested info).
- For CSV: include column headers and rows in proper table structure.
- For JSON: use key-value pairs with nested structure if needed.
- DO NOT add extra notes, explanations, or markdown – just return raw {output_format} content.

Start extracting:
"""


@app.post("/process-pdf/")
async def process_pdf(
    file: UploadFile = File(...),
    output_format: str = Form("text")  # text | json | csv
):
    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Read bytes and generate content
        file_bytes = pathlib.Path(temp_path).read_bytes()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                EXTRACTION_PROMPT_TEMPLATE
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
