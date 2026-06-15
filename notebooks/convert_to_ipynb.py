import json
import re

with open("Thesis-2_Experimental_Results.py", "r") as f:
    text = f.read()

if '# %%' in text:
    cells = text.split('# %%')
else:
    cells = re.split(r'(?:^|\n)# In\[\d+\]:\n+', text)

nb_cells = []
for cell in cells:
    lines = cell.strip().split('\n')
    if not lines or (len(lines) == 1 and lines[0] == ''):
        continue
    if lines[0].startswith('#!/usr') or lines[0].startswith('# coding'):
        continue
    if lines[0].startswith('[markdown]'):
        source = [line.replace('# ', '', 1) + ('\n' if i < len(lines[1:]) - 1 else '') for i, line in enumerate(lines[1:])]
        nb_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source
        })
    else:
        source = [line + ('\n' if i < len(lines) - 1 else '') for i, line in enumerate(lines)]
        nb_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        })

nb = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("Thesis-2_Experimental_Results.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Successfully converted Thesis-2_Experimental_Results.py to Thesis-2_Experimental_Results.ipynb")
