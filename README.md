# TelecomCustomerServiceCallAnalysis
Employee Call Dashboard
Overview
The Employee Call Dashboard is an interactive web application built with Streamlit that provides automated analysis of employee phone calls. It leverages state-of-the-art speech recognition and natural language processing models to transcribe calls, analyze the sentiment of conversations, and infer emotional cues based on speech patterns. This tool is designed to help managers and HR professionals monitor call quality, employee performance, and customer interactions efficiently.

Features
Audio Playback: Listen to call recordings directly within the app.

Automatic Transcription: Uses the faster-whisper model to transcribe audio files quickly and accurately.

Sentiment Analysis: Applies a fine-tuned RoBERTa model to classify the sentiment of the conversation into negative, neutral, or positive categories.

Pitch and Speech Rate Analysis: Extracts average pitch and speaking rate from calls using librosa to infer emotional hints such as excitement, sadness, or calmness.

Emotion Interpretation: Combines pitch and speech rate data to provide human-readable insights about the speaker’s emotional state.

Summary Dashboards:

Sentiment distribution across all calls

Trends in pitch and words per second

Pie chart visualization of emotion hints

Overall employee performance evaluation based on sentiment ratios

Multi-employee support: Select from different employee folders and analyze multiple calls per employee.

Cached Model Loading: Efficient loading and reuse of machine learning models to speed up processing.

How It Works
Load Employees: The app scans a directory where employee call recordings are stored (organized by employee folders).

Select Employee: Choose an employee from the dropdown list to view and analyze their calls.

Analyze Calls: For each call recording:

The audio is played back for listening.

The call is transcribed using the Whisper speech-to-text model.

The transcript is analyzed for sentiment using the RoBERTa model.

Pitch and speaking rate are computed to estimate emotional tone.

Combined insights are displayed in an easy-to-understand format.

Summary: Aggregated charts and stats provide a snapshot of overall call sentiment and emotional characteristics, along with a performance evaluation summary.

Technologies Used
Streamlit: For building the interactive web interface.

faster-whisper: For fast and accurate speech-to-text transcription.

Transformers (Hugging Face): For fine-tuned RoBERTa sentiment analysis model.

Librosa: For audio signal processing (pitch extraction, duration).

Pydub: For audio format conversion and preprocessing.

Torch (PyTorch): For running the sentiment analysis model.

Matplotlib & Pandas: For data visualization and summary statistics.
