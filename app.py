import gradio as gr
from pathlib import Path
import tempfile
from pydub import AudioSegment
from faster_whisper import WhisperModel
import random
import re
import os

# -----------------------------
# FFmpeg setup
# -----------------------------
AudioSegment.converter = "ffmpeg"  # make sure ffmpeg is installed

# -----------------------------
# Load Whisper model
# -----------------------------
print("Loading Whisper model (small)...")
model = WhisperModel("small")  # fast, accurate, no API needed

# -----------------------------
# Extract audio from file (video or audio)
# -----------------------------
def extract_audio(input_path: str) -> str:
    """Convert uploaded video/audio to 16kHz mono WAV."""
    output_path = Path(tempfile.gettempdir()) / (Path(input_path).stem + ".wav")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return str(output_path)

# -----------------------------
# Generate MCQs
# -----------------------------
def generate_mcq_from_transcript(transcript, num_questions=3):
    sentences = re.split(r'[.?!]\s+', transcript)
    sentences = [s for s in sentences if len(s.split()) > 3][:num_questions]
    questions = []

    for i, sentence in enumerate(sentences):
        words = [w.strip(".,") for w in sentence.split() if len(w.strip(".,")) > 3]
        if not words:
            continue
        answer = random.choice(words)
        all_words = [
            w.strip(".,") for w in transcript.split()
            if len(w.strip(".,").strip()) > 3 and w.strip(".,") != answer
        ]
        options = random.sample(all_words, min(3, len(all_words)))
        options.append(answer)
        random.shuffle(options)
        questions.append({
            "question": f"{i+1}. Key concept from: '{sentence[:60]}...'",
            "options": options,
            "answer": answer
        })
    return questions

# -----------------------------
# Process video/audio → transcript + quiz
# -----------------------------
def process_media(file_path):
    if not file_path:
        return "Please upload a file.", [], None

    try:
        wav_path = extract_audio(file_path)
        segments, _ = model.transcribe(wav_path)
        transcript = " ".join([seg.text for seg in segments])

        if not transcript.strip():
            return "No transcript generated.", [], None

        quiz = generate_mcq_from_transcript(transcript)
        return transcript, quiz, create_temp_file(transcript, "transcript.txt")

    except Exception as e:
        return f"Error: {e}", [], None

# -----------------------------
# Save transcript temporarily
# -----------------------------
def create_temp_file(text, filename):
    path = Path(tempfile.gettempdir()) / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)

# -----------------------------
# Evaluate quiz answers
# -----------------------------
def evaluate_answers(quiz, *answers):
    correct = 0
    feedback = []
    for i, q in enumerate(quiz):
        if i < len(answers) and answers[i] == q["answer"]:
            feedback.append(f"✅ Q{i+1}: Correct! ({answers[i]})")
            correct += 1
        else:
            feedback.append(f"❌ Q{i+1}: Wrong! Correct: {q['answer']}")
    feedback.append(f"\nFinal Score: {correct}/{len(quiz)}")
    return "\n".join(feedback)

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="🎙️ Lecture Voice-to-Notes (Offline)") as demo:
    gr.Markdown("# 🎙️ Lecture Voice-to-Notes Generator (Offline)")
    gr.Markdown("Upload a **lecture video or audio** → Get transcript + auto MCQs.")

    with gr.Row():
        video_in = gr.Video(label="🎞️ Upload Lecture Video", height=200, width=320)
        audio_in = gr.Audio(label="🎧 Or Upload Audio File", type="filepath")

    transcript_box = gr.Textbox(label="🗣️ Transcript", lines=10)
    download_link = gr.File(label="📥 Download Transcript", visible=False)
    quiz_state = gr.State([])

    start_btn = gr.Button("🧠 Generate Transcript & Quiz")

    question_boxes = []
    answer_radios = []
    for i in range(3):
        q_text = gr.Markdown(visible=False)
        q_opts = gr.Radio(choices=[], label=f"Q{i+1}", visible=False)
        question_boxes.append(q_text)
        answer_radios.append(q_opts)

    submit_btn = gr.Button("✅ Submit Answers")
    feedback_box = gr.Textbox(label="Feedback", lines=8)

    # Step 1 — Generate Transcript + Quiz
    def handle_generate(video_path, audio_path):
        file_path = video_path if video_path else audio_path
        transcript, quiz, file_link = process_media(file_path)
        if not quiz:
            return [transcript] + [gr.update(visible=False)] * 6 + [quiz, gr.update(visible=False)]

        updates = [transcript]
        for i in range(3):
            if i < len(quiz):
                updates.append(gr.update(value=quiz[i]["question"], visible=True))
                updates.append(gr.update(choices=quiz[i]["options"], visible=True))
            else:
                updates.append(gr.update(visible=False))
                updates.append(gr.update(visible=False))
        updates.append(quiz)
        updates.append(gr.update(value=file_link, visible=True))
        return updates

    start_btn.click(
        fn=handle_generate,
        inputs=[video_in, audio_in],
        outputs=[transcript_box]
        + [v for pair in zip(question_boxes, answer_radios) for v in pair]
        + [quiz_state, download_link]
    )

    # Step 2 — Evaluate answers
    submit_btn.click(
        fn=evaluate_answers,
        inputs=[quiz_state] + answer_radios,
        outputs=[feedback_box]
    )

# ✅ Launch with shareable link
demo.launch(share=True)
