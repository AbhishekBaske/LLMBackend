from fastapi import FastAPI
import subprocess
import os
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str  # Request model to validate incoming query

# Function to extract and format the response from the model output
def extract_rag_response(input_string):
    # Split the string into lines
    lines = input_string.splitlines()
    
    # Extract questions and answers
    extracted_lines = []
    for line in lines:
        if line.strip().startswith("Question:") or line.strip().startswith("Answer:"):
            extracted_lines.append(line.strip())
    
    # Join the extracted lines with newlines for formatted output
    formatted_output = "\n".join(extracted_lines)
    return formatted_output

# POST endpoint to handle the query
@app.post("/query")
async def query(request: QueryRequest):
    query = request.query  # Extract query from the request model
    
    # Command to run the model with a query
    command = [
        "modal",            # Main command to invoke the model
        "run", 
        "api.py",           # Action to perform
        f"--query={query}"  # The query to pass
    ]

    # Run the subprocess and capture the output
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard error
        text=True                # Ensure output is returned as a string
    )

    # Get the outputs
    response = result.stdout.strip()
    error = result.stderr.strip()

    # If the subprocess encountered an error, return it immediately
    if result.returncode != 0:
        return {"error": f"Subprocess failed: {error}"}

    # Extract relevant response
    output = extract_rag_response(response)
    
    # Generate a unique filename using timestamp
    filename = f"output.txt"
    
    # Write output to file
    with open(filename, "w") as file:
        file.write(output)

    return {"output": output, "error": error, "filename": filename}

# GET endpoint to retrieve the output file content
@app.get("/output/")
async def get_output():
    file_path = f"output.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            output = file.read()
        return {"output": output}
    else:
        return {"error": "File not found"}
