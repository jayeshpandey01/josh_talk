You are provided with ~10 hours of Hindi ASR training data (here) in the format shown below
(audio + transcription metadata)
Important Note: The Url’s mentioned above in the the question and further questions might
not work, PFB the instructions for modifying the same
Instructions to access the data :this is the example of a new transcription URL
https://storage.googleapis.com/upload_goai/967179/825780_transcription.json, the recording
and metadata follows the same format, please modify the other URL's while processing the data
Dataset Schema Description
● user_id – Identifier for the speaker/user associated with the audio (anonymized). ●
recording_id – Unique identifier for the specific audio recording within the dataset. ●
language – Language label of the audio (e.g., "hi" for Hindi).
● duration – Duration of the audio recording (in seconds). Useful for filtering or batching.
● rec_url_gcp – URL link to the raw audio file stored on cloud (e.g., Google Cloud
Storage). This is the main audio input for training/evaluation.
● transcription_url – URL to the ground-truth transcription text corresponding to the
audio file. This is the label to be used for fine-tuning.
● metadata_url – URL to additional metadata about the recording (may include device
type, noise level, accents, or collection conditions). Optional for training, but can help in
analysis.
Your Task
a) Preprocess the dataset and share what you did to process the data and make it ready
for- [ ] Implementation [/]
    - [/] Setup Whisper model and load dataset
    - [/] Implement Number Normalization module
    - [x] Implement English Word Detection module (in Devanagari)
    - [x] Create processing pipeline
b) Fine-tune Whisper-small on this dataset and evaluate both the pretrained Whisper-small
baseline and your fine-tuned model on the Hindi portion of the FLEURS test dataset.
c) Report the Word Error Rate (WER) in a structured table format. Here
d) Systematically sample at least 25 utterances where your fine-tuned model still produces
errors. Describe your sampling strategy (e.g., every Nth error, stratified by severity). Do not
cherry-pick examples.
e) Build an error taxonomy from what you observe. Categories should emerge from the
data itself. For each category, provide 3–5 concrete examples showing: the reference
transcript, your model's output, and your reasoning about the cause of the error.
f) For your top 3 most frequent error types, propose a specific, actionable fix. Sometimes
collecting more data is not sufficient.

g) Implement at least one of your proposed fixes within the assignment timeframe. Show
before/after results on a targeted subset of your error examples.