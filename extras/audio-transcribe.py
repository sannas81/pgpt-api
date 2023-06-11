#!/usr/bin/env python3
import whisper
import pandas as pd

model = whisper.load_model("base",download_root="./models/whisper")

files = ["podcast_ideacast.mp3"]
txt_files=[""]
txt_files[0] = files[0].replace("mp3","txt")

result = model.transcribe(audio=files[0],language="en")


transcription = pd.DataFrame(result['segments'])

def chunk_clips(transcription, clip_size):
    texts = []
    sources = []
    for i in range(0, len(transcription),clip_size):
        clip_df = transcription.iloc[i:i+clip_size,:]
        text = " ".join(clip_df['text'].to_list())
        source = str(round(clip_df.iloc[0]['start']/60,2))+ " - " + str(round(clip_df.iloc[-1]['end']/60,2))+ " min"
        texts.append(text)
        sources.append(source)
    return [texts,sources]

chunks = chunk_clips(transcription, 50)

f= open(txt_files[0],"w")

for i in range(len(chunks[0])):
    f.write(chunks[1][i]+" : ")
    f.write(chunks[0][i]+"\n")
f.close()

chunks = None
transcription = None
result = None
model = None