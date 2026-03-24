Raw ASR output from Hindi conversations is messy, numbers come out as words, English words spoken in 
conversation are not always identified or handled correctly etc. Before ASR output is usable for any 
downstream task, it needs to be cleaned up. This question asks you to build a cleanup pipeline with two 
specific operations and carefully check where each one helps and where it makes things worse. 
Keep in mind our transcription guideline: English words spoken in the conversation are transcribed in 
Devanagari script. For example, "computer" spoken in English should appear as "कं प्यूटर." The Devanagari 
transcription counts as the correct spelling, not an error. 
Data  
Using the same ~10-hour dataset, generate raw ASR transcripts by running the pretrained whisper-small 
(before your Q1 fine-tuning) on the audio segments. Pair each raw ASR output with the human reference 
transcription from the dataset's JSON files. 
Your Task  
Build a pipeline that takes raw ASR output and performs the following operations: 
a) Number Normalization 
Convert spoken Hindi number words into digits. 
● Simple cases: दो → 2, दस → 10, सौ → 100 
● Compound numbers: तीन सौ चौवन → 354, पच्चीस → 25, एक हज़ार → 1000 
● Edge cases: how do you handle numbers used in idioms or phrases where conversion would 
be wrong? (e.g., "दो-चार बातें" should probably stay as-is, not become "2-4 बातें") 
Provide 4-5 before/after examples from your actual data showing correct conversions, and 2-3 
examples of tricky edge cases where you had to make a judgment call. Explain your reasoning for 
each edge case. 
b) English Word Detection 
Identify which words in the Hindi transcript are actually English words spoken in the conversation. 
This is important because: 
● English words need different handling in downstream processing 
● They may need script normalization (Roman ↔ Devanagari) 
● They are common in real Hindi conversation ("मेरा interview अच्छा गया", "ये problem solve नहीं 
हो रहा") 
For each transcript, output a tagged version where English words are marked. For example: 
● Input: "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मल गई" 
● Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मल गई" 
