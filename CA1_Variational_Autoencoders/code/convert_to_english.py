# This script will help identify cells but conversion will be done manually with edit_notebook
# For now, just list what needs conversion
import json

with open('code.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print("Cells that need English conversion:")
print("Note: Actual conversion should be done via edit_notebook tool")
