import streamlit as st
from faster_whisper import WhisperModel
import librosa
import numpy as np
import torchaudio
import os
import joblib
import io
from pydub import AudioSegment
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt



@st.cache_resource
def load_sentiment_model():
    model_dir = Path("C:/Users/jeeva/PycharmProjects/PythonProject1/finetuned-roberta-telecom")
    import os
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print("model_dir", model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

sentiment_model, sentiment_tokenizer = load_sentiment_model()

def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[prediction]


def analyze_pitch_and_rate(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    # 1. Extract pitch using librosa.yin
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    pitch = pitch[pitch > 0]  # remove invalids
    avg_pitch = np.mean(pitch) if len(pitch) > 0 else 0

    # 2. Speaking rate (words per second)
    # Use whisper transcript word count
    transcript = transcribe_audio_faster_whisper(audio_path)
    word_count = len(transcript.split())
    duration = librosa.get_duration(y=y, sr=sr)
    words_per_sec = word_count / duration if duration > 0 else 0

    return {
        "avg_pitch": avg_pitch,
        "duration": duration,
        "words_per_sec": words_per_sec,
        "word_count": word_count,
        "transcript": transcript
    }

def interpret_emotion_from_audio(pitch, wps):
    result = []

    if pitch > 180:
        result.append("Elevated pitch (possible excitement or anger)")
    elif pitch < 100:
        result.append("Low pitch (possible sadness or tiredness)")

    if wps > 3.0:
        result.append("Fast speech (possibly rushed or upset)")
    elif wps < 1.5:
        result.append("Slow speech (possibly calm or thoughtful)")

    if not result:
        return "Normal tone and pace"
    return ", ".join(result)


def ensure_wav_16k(audio_path: str) -> io.BytesIO:
    # Load and convert audio
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Export to memory buffer instead of a file
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav", codec="pcm_s16le")
    wav_io.seek(0)  # Reset cursor to the beginning of the buffer

    return wav_io


# Load the model once (choose 'medium' or 'large-v2' for better accuracy)
@st.cache_resource
def load_model():
   model = WhisperModel("medium", device="cpu", compute_type="int8")  # Or use 'cuda' if you have GPU
   return model
model = load_model()
@st.cache_data(show_spinner=False)
def transcribe_audio_faster_whisper(file_path: str, language: str = "en") -> str:
    """
    Transcribes an audio file using faster-whisper.

    Args:
        file_path (str): Path to audio file.
        language (str): Language code like 'en', 'ms', etc.

    Returns:
        str: Transcribed text.
    """
    segments, info = model.transcribe(file_path, language=language, beam_size=5)

    # Combine all segments into one text block
    transcription = " ".join([segment.text for segment in segments])
    return transcription


# Path to where you store your employee folders
BASE_PATH = "C:\\Users\\jeeva\\OneDrive\\Desktop\\Recordings"

st.title("📞 Employee Call Dashboard")

# List all employee folders
employees = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]

if not employees:
    st.warning("No employee folders found.")
else:
    selected_employee = st.selectbox("Select an Employee", employees)

    if selected_employee:
        st.header(f"Calls for {selected_employee}")

        emp_folder = os.path.join(BASE_PATH, selected_employee)
        audio_files = [f for f in os.listdir(emp_folder) if f.lower().endswith(('.wav', '.mp3', '.m4a'))]

        if not audio_files:
            st.info("No audio files found.")
        else:
            sentiment_counts = Counter()
            emotion_summary = []
            call_stats = []
            for audio in sorted(audio_files):
                audio_path = os.path.join(emp_folder, audio)
                st.subheader(audio)

                # Display audio player
                st.audio(audio_path)

                # Transcription
                with st.spinner("Transcribing..."):
                    transcript = transcribe_audio_faster_whisper(audio_path, language="en")
                    st.write("**Transcript:**")
                    st.write(transcript)
                with st.spinner("Roberta model..."):
                    # Sentiment Prediction
                    sentiment = predict_sentiment(transcript)
                    st.write("**Sentiment (RoBERTa):**", sentiment)
                with st.spinner("Analyzing emotion..."):
                    analysis = analyze_pitch_and_rate(audio_path)
                    emotion_hint = interpret_emotion_from_audio(
                        analysis["avg_pitch"], analysis["words_per_sec"]
                    )

                    st.write("**Pitch & Speech Analysis:**")
                    st.write(f"- Avg Pitch: {analysis['avg_pitch']:.2f} Hz")
                    st.write(f"- Speaking Rate: {analysis['words_per_sec']:.2f} words/sec")
                    st.write(f"- Emotion hint: {emotion_hint}")
                st.markdown("### 🔍 Combined Emotion Summary")
                summary = []

                if sentiment == "negative":
                    summary.append("The content of the call was negative.")
                elif sentiment == "positive":
                    summary.append("The conversation had a positive sentiment.")
                else:
                    summary.append("The tone of the conversation was neutral.")

                summary.append(f"Speech pattern suggests: {emotion_hint}.")

                st.info(" ".join(summary))
                # Save to summary stats
                sentiment_counts[sentiment] += 1
                emotion_summary.append(emotion_hint)

                call_stats.append({
                    "filename": audio,
                    "pitch": analysis["avg_pitch"],
                    "wps": analysis["words_per_sec"],
                    "sentiment": sentiment,
                    "emotion": emotion_hint
                })
            st.header("📊 Overall Performance Summary")

            # Convert to DataFrame
            df_stats = pd.DataFrame(call_stats)

            # 1. Sentiment Distribution Bar Chart
            st.subheader("Sentiment Distribution")
            sent_df = pd.DataFrame(sentiment_counts.items(), columns=["Sentiment", "Count"])
            st.bar_chart(sent_df.set_index("Sentiment"))
            # 2. Performance Evaluation Summary
            st.subheader("🧠 Performance Analysis")

            num_calls = len(audio_files)
            neg_ratio = sentiment_counts["negative"] / num_calls if num_calls else 0
            pos_ratio = sentiment_counts["positive"] / num_calls if num_calls else 0

            performance = "Good" if pos_ratio > 0.5 else "Needs Improvement" if neg_ratio > 0.3 else "Average"

            st.success(f"Employee Performance: **{performance}**")
            st.markdown(f"- Positive Calls: {pos_ratio:.0%}")
            st.markdown(f"- Negative Calls: {neg_ratio:.0%}")
            st.markdown(f"- Total Calls Analyzed: {num_calls}")


