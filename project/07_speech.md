# 7. Speech test

**Task:** the person looks at the "Cookie Theft" picture (a kitchen scene) and describes everything they see.
People with Alzheimer's often pause, repeat words and hesitate more.

**Output:** *flagged* or *not flagged*. Code: `app/speech.py`, `speech/whisper_features.py` · Model: `models/speech_whisper.onnx`

## Method, step by step

1. **Record.** The browser records the microphone and sends the audio to the server.
2. **Speech to text with timing.** **Whisper** (OpenAI's open speech-recognition model, "small" size, run locally through `faster-whisper`) turns the audio into words **with a start and end time for each word**. A short hint phrase ("Um, uh, so, er, I think...") encourages it to keep hesitation words.
3. **Five numbers** (`speech/whisper_features.py`), all per 10 seconds of speech:

   | Number | Meaning |
   |---|---|
   | `fillers` | Words like um, uh, er |
   | `mhm_hm` | Sounds like mm-hmm, hmm |
   | `repetition` | The same word said twice in a row ("the the") |
   | `pause` | A gap of 0.5 s or more between words |
   | `word_rate` | How many real words are spoken |

4. **Model.** A small model (`speech_whisper.onnx`) takes the five numbers and gives the probability of "looks like the Alzheimer's group". At 0.5 or above it is flagged.
5. **Quality gate.** If fewer than 20 words were heard, the result is "not run" (too little speech to judge). That is a data-quality rule, not a medical threshold.

The transcript is shown back to the person on the page but is **never stored**.

## Where the training data came from

ADReSS Cookie Theft recordings (108 people). Published methods use human-typed transcripts (which mark every um and pause). A live app can't have those, so we use only the five numbers that can be recovered from Whisper's words and timing.

## Honest limits

- On recordings transcribed by Whisper small, 5-fold cross-validation: balanced accuracy **0.639 (95% CI 0.546–0.731)**, 108 people. (On human transcripts it was higher, 0.759, but that can't be produced live.)
- Whisper tends to drop hesitations, which weakens exactly the signal we look for.
- The live app uses a different Whisper build, a laptop microphone, and a redrawn "Cookie Theft" picture, none of which is validated.
