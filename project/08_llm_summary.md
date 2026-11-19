# 8. The written summary (LLM)

After the four tests, the page shows a short plain-English note for the doctor.
Code: `app/summary.py`, `app/combine.py`

## Two small pieces

### a) Counting flags (`combine.py`)
It simply lists each test as *flagged*, *not flagged* or *not completed*, and states, for example,
"2 of 4 completed screens flagged." There is **no averaging, no weighting and no combined score** — the four tests
are different and independent.

### b) The language model (`summary.py`)
An **LLM** (large language model) is a program that writes text. Ours is **Llama 3.1 (8B)**, run on the laptop
through **Ollama**, so patient results are not sent to any online service.

## How it stays safe

The model is only allowed to *rephrase* the results. Several guards check it:

1. **Fixed input.** It is given only the structured results (labels and measured values).
2. **Fixed format.** It must reply in a set JSON shape: one short paragraph each for clock, motor, eye and speech, plus an overall line.
3. **Banned words.** If the reply contains words such as *diagnosis, stage, early-stage, mild, moderate, severe, severity, prognosis, likelihood, risk score, overall accuracy*, the reply is thrown away.
4. **No invented numbers.** Every number in the reply must already appear in the results. Otherwise the reply is thrown away.
5. **Fixed overall line.** The "x of y flagged" line is always the program's own text, never the model's.
6. **Fallback template.** If the model is not installed, too slow, or breaks any rule, the app uses a plain fixed template instead. The page says why the template was used.

The template is also used when fewer than four tests were completed.

## Setting it up

- Install **Ollama** and run `ollama pull llama3.1:8b` (about 5 GB). `setup_new_pc.bat` does both.
- Without Ollama the app still works; the summary just uses the template text.
