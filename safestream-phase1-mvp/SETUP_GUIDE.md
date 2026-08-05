# SafeStream Phase 1 MVP — Setup Guide

Goal by Thursday: camera preview on a real phone, with detected faces
visibly blurred in real time. No streaming yet.

## Who does what today (Tuesday)

- **Person A**: Steps 1-3 below (Android Studio project + camera preview)
- **Person B**: `model-prep/prepare_model.py` (needs a laptop with Python, not
  the phone) — produces the `.tflite` file Person A/C will need tomorrow
- **Person C**: Start the progress report using what's already written in
  `report_14_merged.pdf` (architecture, objectives, literature survey are
  already done — the report just needs a new "Progress" section)
- **You**: unblock whoever gets stuck; review PRs/commits at day's end

## Step 1: Create the Android Studio project

1. Android Studio → New Project → **Empty Views Activity** (not Compose —
   simpler for this timeline)
2. Language: **Kotlin**. Minimum SDK: **API 26** (Android 8.0) — needed for
   CameraX + hardware acceleration APIs.
3. Package name: `com.safestream.app` (matches the code files below)

## Step 2: Add dependencies

Copy the contents of `app-kotlin/build.gradle.kts.snippet` into your
`app/build.gradle.kts`. Sync Gradle.

For OpenCV specifically:
1. Download the Android SDK from https://opencv.org/releases/
2. Unzip it
3. Android Studio → File → New → Import Module → point at the unzipped
   `sdk` folder
4. Make sure `settings.gradle.kts` includes `include(":opencv")`

## Step 3: Camera permission

Add to `AndroidManifest.xml`, above `<application>`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera.any" />
```

## Step 4: Drop in the code files

- `app-kotlin/MainActivity.kt` → `app/src/main/java/com/safestream/app/MainActivity.kt`
- `app-kotlin/YoloDetector.kt` → same folder
- `app-kotlin/BlurOverlayView.kt` → same folder
- `app-kotlin/activity_main.xml` → `app/src/main/res/layout/activity_main.xml`

## Step 5: Add the model (once Person B finishes model-prep)

Copy `yolov8n-face.tflite` into `app/src/main/assets/` (create the `assets`
folder if it doesn't exist: right-click `app/src/main` → New → Directory →
`assets`).

**Before trusting the app to run correctly**, run
`model-prep/verify_tflite_model.py` on the model and compare its printed
output shape against the comments in `YoloDetector.kt` (`OUTPUT_COLS`,
`INPUT_SIZE`). If they don't match, the detection boxes will be nonsense
and you'll waste hours debugging the wrong file.

## Step 6: Run it

Plug in a real Android phone (emulators can't do a live camera feed
usefully). Run the app. You should see:
1. Your camera feed
2. A blurred box wherever a face is detected

## Known rough edges (budget time for these)

- **YOLOv8 TFLite output shape varies by ultralytics version.** This is the
  #1 thing that breaks. Always verify before debugging the Kotlin side.
- **Front camera rotation**: `toBitmap()` in `MainActivity.kt` handles the
  common 90°/270° rotation case, but test on your actual device — some
  OEMs report rotation differently.
- **First build with OpenCV imported is slow** (native libs). Don't panic
  if the first Gradle sync takes several minutes.
- I can't compile-test Kotlin/Android code from here (no Android SDK in my
  environment) — treat this as "should work, verify on device," not
  guaranteed bug-free. Come back with the actual error if something breaks
  and I'll debug it with you.

## What to show the panel Thursday

- The app running live on a phone with visible face blurring
- A short screen-recorded backup video (record this Wednesday night in case
  something glitches during the live demo)
- The progress report noting: MVP architecture built (camera → detection →
  blur → display), streaming/RTMPS and identity-whitelisting are the next
  milestones per the roadmap
