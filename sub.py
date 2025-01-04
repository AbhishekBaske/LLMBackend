import subprocess
import os

query = "What is gassy mines?"    
# Command to run the model with a query
command = [
    "modal",           # The main command to invoke the model
    "run",
    "api.py",             # The action to perform
    f"--query={query}"   # The query to pass
]

# Run the subprocess and capture the output
result = subprocess.run(
    command,
    stdout=subprocess.PIPE,  # Capture standard output
    stderr=subprocess.PIPE,  # Capture standard error
    text=True                # Ensure output is returned as a string
)

# Get the outputs
output = result.stdout.strip()
error = result.stderr.strip()




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

# Call the function and get the result
rag_response = extract_rag_response(output)

# Save the result to a file or print it
print("Formatted RAG Response:")
print(rag_response)

# Save to file
output_file_path = "output.txt"
with open(output_file_path, "w") as file:
    file.write(rag_response)
