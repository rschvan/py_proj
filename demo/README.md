# PyPathfinder demo video

Records a silent ~10-minute walkthrough of the app and writes the narration to read over it.

## Record

```
venv/Scripts/python.exe -m pip install playwright      # once
venv/Scripts/python.exe -m playwright install chromium # once
venv/Scripts/python.exe demo/record_demo.py            # full recording, about 11 minutes
venv/Scripts/python.exe demo/record_demo.py --fast     # quick test run, about 3 minutes
```

The recorder starts its own copy of the app (port 8599), so your usual app can keep running.
Results go to `demo/output/` (not committed):

- `pypf_demo.webm`: the video, 1600×900. YouTube accepts .webm directly.
- `demo_script.md`: the narration for each scene with its start time, plus YouTube chapter
  timestamps to paste into the video description.

## Change the demo

Everything is in `scenes.py`: each scene has a title (a YouTube chapter), its narration, and
its actions. A scene lasts until its narration would have been read at about 145 words a
minute, so edit the words and the timing follows. Re-record after changing the app or the scenes.

## Add your voice

Play the video and read each section of `demo_script.md` as it comes up, recording with any
voice-over tool, for example Clipchamp (included with Windows 11): import the video, record a
voice-over, and export as .mp4. Or upload the video and narration separately to an editor of
your choice.
