In ASR evaluation, comparing model output against a single, rigid 
Ground Truth string unfairly penalizes valid transcriptions. Speech often has multiple 
correct written representations. A Lattice addresses this by replacing a flat string 
with a sequential list of "bins." Each bin represents a specific alignment position and 
contains all valid lexical, phonetical, and spelling variations for that point in the 
audio. Here(datasets), you are given transcriptions from five ASR models for the same audio 
and a human reference, which may contain errors.  
Example: If the spoken audio is "उसने चौदह कताबें खरीदीं" (He bought 14 books), a 
rigid reference transcript might just be ["उसने", "चौदह", "कताबें", "खरीदीं"]. A 
lattice representation captures valid alternatives (numbers vs. words, spelling 
variations, and lexical synonyms) and groups them sequentially: [["उसने"], 
["चौदह", "14"], ["कताबें", "कताबे", "पुस्तकें"], ["खरीदीं", "खरीदी"]] 
Design an approach (theory + pseudocode/code) to:  
● Construct a lattice that captures all valid transcription alternatives from the model 
outputs.  
● Handle insertions, deletions, and substitutions in a way that does not unfairly penalize 
models when the reference is wrong.  
● Decide when to trust model agreement over the reference.  
Choose and justify the alignment unit (word / subword / phrase). Then compute WER for each 
model using lattice based transcription and model output. Your method should reduce WER 
for models that were unfairly penalized and keep it unchanged for the others.