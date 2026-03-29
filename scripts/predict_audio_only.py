
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tribe_helpers import check_tribev2, load_model


def build_audio_only_events(video_path: str) -> pd.DataFrame:
    from neuralset.events.transforms import ExtractAudioFromVideo, ChunkEvents
    from neuralset.events.utils import standardize_events

    event = {
        "type": "Video",
        "filepath": str(Path(video_path).resolve()),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    events = standardize_events(pd.DataFrame([event]))

    for transform in [
        ExtractAudioFromVideo(),
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30),
    ]:
        events = transform(events)

    return standardize_events(events)


def main():
    parser = argparse.ArgumentParser(description="Predict brain responses (audio+video only, no WhisperX)")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, default="./results/tribe_predictions")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    check_tribev2()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stimulus_name = Path(args.video).stem

    model = load_model(cache_dir=args.cache_dir, device=args.device)

    events_df = build_audio_only_events(args.video)
    print(f"Events: {len(events_df)} rows")
    print(events_df[["type", "start", "duration"]].to_string(index=False))

    preds, segments = model.predict(events=events_df)
    print(f"Predictions shape: {preds.shape}")

    np.save(output_dir / f"{stimulus_name}_predictions.npy", preds)
    with open(output_dir / f"{stimulus_name}_segments.pkl", "wb") as f:
        pickle.dump(segments, f)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
