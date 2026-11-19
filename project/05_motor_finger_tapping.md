# 5. Finger tapping test (the "motor" test)

**Task:** the person taps one button as fast as they can for **30 seconds**, with one hand (the dominant hand).
Movement speed and rhythm are affected by brain health, and age.

**Output:** *flagged* or *not flagged*. Code: `app/motor.py`, `motor/reference.py` · Data file: `models/motor_reference.json`

**This test uses no AI model.** It uses simple rules and a table of normal values.

## Method, step by step

1. **Record tap times.** The browser notes the time of every tap (in seconds from the start). The person's age comes from the questionnaire.
2. **Calculate three numbers** from the taps:

   | Number | Meaning |
   |---|---|
   | `tps` | Taps per second (speed) |
   | `cv` | How uneven the gaps between taps are (std ÷ mean of the gaps). Higher = shakier rhythm |
   | `fatigue` | Taps in the last 10 s ÷ taps in the first 10 s (do they slow down?) |

   If there are fewer than 10 taps, the test is "not completed".
3. **Compare with healthy people of the same age.** A published reference (Lima et al. 2024: 176 healthy adults aged 18–74, 30-second smartphone tapping) was used to draw a line for each number against age. For this person's age we know what is "expected", and how far they are above or below it.
4. **Cut-offs** (fixed in advance, not tuned): a result is "outside the range" if speed or fatigue falls below the 5th percentile of the healthy group, or unevenness rises above the 95th percentile.
5. **Flag rule**
   - Tapping **faster** than the healthy range is **never** flagged.
   - Tapping clearly **slower** than expected for the age is flagged on its own.
   - Otherwise it is flagged when at least **2 of the 3** numbers are outside the range.

   (This rule was chosen by the project owner on 2026-10-19; the original needed 2 of 3 always.)

The page then shows taps per second, rhythm variability, last-10 s / first-10 s ratio, and which numbers were outside.

## Honest limits

- It does **not** say "Alzheimer's". It says the tapping is outside the usual range for that age.
- **No accuracy has been measured**: no patient data was used. It has not been tested on people with the disease.
- The reference group is young (median age 30), used a phone touchscreen, and has only about 15 adults over 50. The people we want to screen are older and use a different device.
- Ages outside 18–74 are treated as 18 or 74 (no guessing beyond the data).
