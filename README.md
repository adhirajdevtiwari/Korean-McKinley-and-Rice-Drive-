# Resume Matcher

This project is a Python-based Resume Matching tool that pairs candidate resumes with Job Descriptions (JDs) by comparing their respective skill sets. It uses Natural Language Processing (NLP) concepts like TF-IDF and Cosine Similarity to find the most suitable candidates for a given role.

## What It Does

1. **Skill Normalization**: Processes raw skill strings from resumes and JDs, correcting typos and mapping aliases (e.g., `pyhton` to `python`, `reacts` to `react`, `machinelearning` to `machine_learning`) using a predefined dictionary.
2. **TF-IDF Vectorization**: Calculates the Term Frequency-Inverse Document Frequency (TF-IDF) for candidate skills. This ensures that rare, specialized skills are weighted more heavily than extremely common ones.
3. **JD Binary Vectors**: Converts Job Descriptions into binary vectors based on the required skills.
4. **Cosine Similarity**: Computes the cosine similarity between the Job Description vectors and the candidate resume vectors to generate a match score.
5. **Top Candidates**: Ranks and outputs the top 3 best-matching candidates for each Job Description along with their similarity score.

## Output

Below is a screenshot of the script in action, showing the top matching candidates and their scores for three different engineering roles:

![Output Terminal](output.png)

*(Note: Please ensure the screenshot is saved as `output.png` in this directory to view it here).*

## How to Run

1. Ensure you have Python installed.
2. Clone this repository.
3. Run the script from your terminal:
   ```bash
   python main.py
   ```
