import os
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parents[1])
DATA_DIR = f"{BASE_DIR}/data"
OUTPUT_DIR = f"{BASE_DIR}/outputs"

def set_zeroshot_paths(dataset_name = "kym", output_suffix = ""):
    DATASET_DIR = f"{DATA_DIR}/{dataset_name}"
    IMAGES_DIR = f"{DATASET_DIR}/images"
    DATA_JSON_PATH = f"{DATASET_DIR}/{dataset_name}_eval.json"
    
    # Path for model predictions
    ZS_DIR = f"{OUTPUT_DIR}/{dataset_name}/zeroshot"
    if not os.path.exists(ZS_DIR): os.makedirs(ZS_DIR)
    OUTPUT_JSON_PATH = f"{ZS_DIR}/pred_{output_suffix}.json"
    
    # Evaluation Paths
    EVAL_RESULTS_PATH = f"{ZS_DIR}/eval_{output_suffix}_details.json"
    EVAL_AVG_PATH = f"{ZS_DIR}/eval_{output_suffix}_avg.json"
    
    return DATASET_DIR, IMAGES_DIR, DATA_JSON_PATH, OUTPUT_JSON_PATH, EVAL_RESULTS_PATH, EVAL_AVG_PATH

def set_retrieve_paths(dataset_name, answer_suffix, output_suffix, selected_model):
    DATASET_DIR = f"{DATA_DIR}/{dataset_name}"
    IMAGES_DIR = f"{DATASET_DIR}/images"
    DATA_JSON_PATH = f"{DATASET_DIR}/{dataset_name}_eval.json"
    
    OUT_DIR = f"{OUTPUT_DIR}/{dataset_name}"
    
    if selected_model == "QWEN3-32B":
        folder_name = ""
    elif selected_model == "GEMMA3-27B":
        folder_name = "/gemma"
    RETRIEVED_JSON_PATH = os.path.join(OUT_DIR, f"wst{folder_name}/wst_fulltext.json")
    ANSWER_JSON_PATH = os.path.join(OUT_DIR, f"answers{folder_name}/answers_{answer_suffix}.json")
    STATEMENT_JSON_PATH = os.path.join(OUT_DIR, f"stat{folder_name}/statements_{output_suffix}.json")
    
    # Evaluation Paths
    EVAL_DIR = f"{OUT_DIR}/eval"
    EVAL_RESULTS_PATH = f"{EVAL_DIR}{folder_name}/eval_{output_suffix}_details.json"
    EVAL_AVG_PATH = f"{EVAL_DIR}{folder_name}/eval_{output_suffix}_avg.json"
    return IMAGES_DIR, DATA_JSON_PATH, \
        OUT_DIR, RETRIEVED_JSON_PATH, ANSWER_JSON_PATH, STATEMENT_JSON_PATH, \
        EVAL_RESULTS_PATH, EVAL_AVG_PATH


# Answer generation SETTINGS
MAX_NEW_TOKENS_ANSWER = 220
MAX_NEW_TOKENS_STATEMENT = 128
DEBUG = True
SAVE_EVERY_MEME = 1

CHUNK_SIZE_CHARS = 128
TOP_K_CHUNKS = 10
MAX_USED_CHUNK_IDS = 10

CHUNK_SIZE_WORDS = 300
OVERLAP_WORDS = 30

# Prompts
ANSWER_PROMPT = r"""
You are answering a question using only the provided retrieved documents.

The retrieved documents were specifically retrieved for this question.

Your goal:
Produce the strongest evidence-grounded answer possible.

Important:
- Use ONLY the provided retrieved documents.
- Do NOT use outside knowledge, memory, assumptions, or unsupported claims.
- If the documents support a full answer, give a concise full answer.
- If the documents support only part of the question, give the best partial evidence-grounded answer.
- If the question asks about significance, implication, symbolism, connection, or why something is presented in a certain way, give a short evidence-grounded interpretation only if it follows directly from the retrieved documents.
- You may combine multiple retrieved facts into one concise answer.
- Preserve important entities and phrases from the question in the answer.
- Keep the answer concise and factual.
- Do NOT mention the documents, chunk numbers, or the rules.
- Only output exactly "No answer can be found." if the retrieved documents contain no relevant information at all.

Return JSON ONLY in this format:
{{
  "answer": "...",
  "used_chunk_ids": [1, 2]
}}

Question:
{question}

Retrieved documents:
{evidence_block}
""".strip()

QA_TO_STAT_DEMOS = """
You are a expert writer. Given a question ([QUES]) and its answer [ANS], your goal is to convert the QA pair into a statement [STAT].

Below are some examples:

[QUES]: Who is Ilhan Omar?
[ANS]: Ilhan Omar is a U.S. Representative for Minnesota's 5th Congressional District, which includes Minneapolis and surrounding suburbs.
[STAT]: Ilhan Omar is a U.S. Representative for Minnesota's 5th Congressional District, including Minneapolis and surrounding suburbs.

[QUES]: Why is Ilhan Omar's statement 'I hate Trump' being paired with the response 'Most Terrorists do'?
[ANS]: The pairing of Ilhan Omar's statement 'I hate Trump' with the response 'Most Terrorists do' appears to be a strategic effort to frame her as sympathetic to terrorism or extremist ideologies.
[STAT]: Ilhan Omar's statement 'I hate Trump' is paired with the response 'Most Terrorists do' to frame her as sympathetic to terrorism or extremist ideologies.

[QUES]: What does the phrase 'It affects virtually nobody. It’s an amazing thing.' refer to?
[ANS]: The phrase 'It affects virtually nobody. It’s an amazing thing.' refers to the coronavirus.
[STAT]: The phrase 'It affects virtually nobody. It’s an amazing thing.' refers to the coronavirus.

Please convert the QA pair below into its statement:
""".strip()


