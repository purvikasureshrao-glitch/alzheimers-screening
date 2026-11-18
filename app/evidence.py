"""Each modality's own measured evidence, copied from FINDINGS.md (numbers are not recomputed here). Update when FINDINGS changes."""

EVIDENCE = {
    "clock": ("NHATS test split, 1089 people, grouped by person: balanced accuracy 0.654 (95% CI 0.605-0.707), AUC 0.707 "
              "(0.631-0.780). It flags about a third of people while 4.8% have dementia (FINDINGS 3d)."),
    "motor": ("No accuracy exists. Compared with 176 healthy adults aged 18-74 (Lima et al. 2024), adjusted for age; "
              "no patient data was used (FINDINGS 10b)."),
    "eye": ("Pilot on ADEM_TEST, 20 people, leave-one-subject-out: balanced accuracy 0.300 (95% CI 0.101-0.505). "
            "The interval includes 0.5: no detectable signal (FINDINGS 8b)."),
    "speech": ("On ADReSS Cookie Theft recordings transcribed by Whisper small, 5-fold CV, 108 people: balanced accuracy "
               "0.639 (95% CI 0.546-0.731). With Whisper medium 0.556 (0.460-0.650), which includes chance. On human-written transcripts "
               "it was 0.759 (0.669-0.841) but those cannot be produced live (FINDINGS 9, 9b, 9c)."),
}

CAVEAT = {
    "clock": "Trained on NHATS scanned drawings; photos of paper drawings, and drawings made with a mouse on screen, are unvalidated domain shifts. The cropping step did not learn well (FINDINGS 3e).",
    "motor": "Not an Alzheimer's detector. The reference group is younger (median age 30), used a touchscreen, and contains few adults over 50.",
    "eye": ("Live webcam setup is unvalidated. The model was trained on a ~333 Hz research tracker (about 0.3 s per task, "
            "unknown screen units); a ~30 Hz webcam gives about 9 gaze points per task and cannot reproduce the saccade or left/right-eye "
            "features. The pilot is too small to show any effect (FINDINGS 8c)."),
    "speech": ("Live microphone setup is unvalidated. Whisper drops most hesitations; the app uses a different Whisper build, a microphone and a "
               "different speaking task than the training recordings, none validated (FINDINGS 9d)."),
}
