"""Each modality's own measured evidence, copied from FINDINGS.md (numbers are not recomputed here). Update when FINDINGS changes."""

EVIDENCE = {
    "clock": ("Not yet available: the Clock model is trained on NHATS scans (Colab run pending). Results will be reported "
              "with balanced accuracy, 95% CI, n and the NHATS test protocol (FINDINGS 3)."),
    "motor": ("No accuracy exists. Compared with 176 healthy adults aged 18-74 (Lima et al. 2024), adjusted for age; "
              "no patient data was used (FINDINGS 10b)."),
    "eye": ("Pilot on ADEM_TEST, 20 people, leave-one-subject-out: balanced accuracy 0.300 (95% CI 0.101-0.505). "
            "The interval includes 0.5: no detectable signal (FINDINGS 8b)."),
    "speech": ("On human-written transcripts, 5-fold CV, 108 people: balanced accuracy 0.759 (95% CI 0.669-0.841). "
               "From a microphone with Whisper it falls to 0.639 (0.546-0.731) and 0.556 (0.460-0.650) (FINDINGS 9, 9b, 9c)."),
}

CAVEAT = {
    "clock": "Trained on NHATS scanned drawings; photos of paper drawings are an unvalidated domain shift.",
    "motor": "Not an Alzheimer's detector. The reference group is younger (median age 30), used a touchscreen, and contains few adults over 50.",
    "eye": "Webcam eye tracking is unvalidated; the pilot is too small to show any effect.",
    "speech": "Live speech screening is research-only until a transcript source that keeps hesitations is validated.",
}
