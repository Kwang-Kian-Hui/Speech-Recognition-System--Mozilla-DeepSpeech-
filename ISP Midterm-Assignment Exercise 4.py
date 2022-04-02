# import necessary modules
import deepspeech
import wave
import numpy as np
import math
from scipy.io import wavfile
import scipy
import librosa

# print menu
print("Select your language-------------Seleccione su idioma-------------Seleziona la tua lingua")
valid = False
language = ""
while(not(valid)):
    username = input("English(EN), Espanol(ES), Italiana(IT): ")
    try:
        # change user input to lower case for easier comparison and check
        username.lower()
        # accepts english and en in any case for english language
        if username == "english" or username == "en":
            # set up language model and score
            language = "en"
            model_file_path = 'models/EN/deepspeech-0.9.3-models.pbmm'
            model = deepspeech.Model(model_file_path)
            scorer_file_path = 'models/EN/deepspeech-0.9.3-models.scorer'
            valid = True
        if username == "espanol" or username == "es":
            language = "es"
            model_file_path = 'models/ES/output_graph_es.pbmm'
            model = deepspeech.Model(model_file_path)
            scorer_file_path = 'models/ES/kenlm_es.scorer'
            valid = True
        if username == "italiana" or username == "it":
            language = "it"
            model_file_path = 'models/IT/output_graph_it.pbmm'
            model = deepspeech.Model(model_file_path)
            scorer_file_path = 'models/IT/kenlm_it.scorer'
            valid = True

        if valid == False:
            print("Please enter a valid language-------------Introduzca un idioma válido-------------Inserisci una lingua valida ")
    except:
        print("Error reading model file.")
    

model.enableExternalScorer(scorer_file_path)
lm_alpha = 0.75
lm_beta = 1.85
model.setScorerAlphaBeta(lm_alpha, lm_beta)
beam_width = 500
model.setBeamWidth(beam_width)

desired_sample_rate = model.sampleRate()

if language == "en":
    filenames = ['Ex4_audio_files/EN/checkin.wav','Ex4_audio_files/EN/checkin_child.wav',
                 'Ex4_audio_files/EN/parents.wav','Ex4_audio_files/EN/parents_child.wav',
                 'Ex4_audio_files/EN/suitcase.wav','Ex4_audio_files/EN/suitcase_child.wav',
                 'Ex4_audio_files/EN/what_time.wav','Ex4_audio_files/EN/what_time_child.wav',
                 'Ex4_audio_files/EN/where.wav','Ex4_audio_files/EN/where_child.wav',
                 'Ex4_audio_files/EN/your_sentence1.wav', 'Ex4_audio_files/EN/your_sentence2.wav']

if language == "es":
    filenames = ['Ex4_audio_files/ES/checkin_es.wav','Ex4_audio_files/ES/parents_es.wav',
                'Ex4_audio_files/ES/suitcase_es.wav','Ex4_audio_files/ES/what_time_es.wav',
                'Ex4_audio_files/ES/where_es.wav']
    
if language == "it":
    filenames = ['Ex4_audio_files/IT/checkin_it.wav','Ex4_audio_files/IT/parents_it.wav',
                'Ex4_audio_files/IT/suitcase_it.wav','Ex4_audio_files/IT/what_time_it.wav',
                'Ex4_audio_files/IT/where_it.wav']
converted_texts = []

# spectral subtraction and low pass filter
m = 1
noise_file = 'Ex4_audio_files/crowd_noise_2.wav'
ats, y_samping_rate = librosa.load(noise_file, sr=desired_sample_rate)
transformed_ats = librosa.stft(ats)
nss = np.abs(transformed_ats)
mns = np.mean(nss, axis=1)

# go through every audio file for the respective language
for filename in filenames:
    file_ats, y_sr = librosa.load(filename, sr=None, mono=True)
    transformed_ats = librosa.stft(file_ats)
    ss = np.abs(transformed_ats)
    angle = np.angle(transformed_ats)
    b = np.exp(1.0j * angle)

    # subtract crowd noise
    subtracted_audio = ss - mns.reshape((mns.shape[0], 1))
    subtracted_audio = subtracted_audio * b
    y = librosa.istft(subtracted_audio)

    # write a new file with the reduced crowd noise
    scipy.io.wavfile.write("Ex4_audio_files/mywav_reduced_noise" + str(m) + ".wav", y_sr, (y * 32768).astype(np.int16))

    # perform low pass filter on reduced crowd noise file
    freq_sampling_rate, data = wavfile.read("Ex4_audio_files/mywav_reduced_noise" + str(m) + ".wav")

    # different cutoff frequency for different languages, explanation in report
    if language == "en" or language == "it":
        cut_off_frequency = 2000.0
    if language == "es":
        cut_off_frequency = 3000.0
    freq_ratio = cut_off_frequency / freq_sampling_rate

    N = int(math.sqrt(0.196201 + freq_ratio**2) / freq_ratio)

    win = np.ones(N)
    win *= 1.0/N
    # low pass filter
    filtered = scipy.signal.lfilter(win, [1], data).astype(np.int16)

    # # retrieve amp width, channel and frame count from reduced noise file
    w = wave.open("Ex4_audio_files/mywav_reduced_noise" + str(m) + ".wav", 'r')
    amp_width = w.getsampwidth()
    n_channels = w.getnchannels()
    n_frames = w.getnframes()
    w.close()

    # write new filtered file
    wav_file = wave.open("Ex4_audio_files/mywav_reduced_noise" + str(m) + "filtered.wav", "w")
    wav_file.setnchannels(1)
    wav_file.setsampwidth(amp_width)
    wav_file.setframerate(desired_sample_rate)
    wav_file.setnframes(n_frames)
    wav_file.writeframes(filtered.tobytes('C'))
    wav_file.close()

    # open newly written file
    w = wave.open("Ex4_audio_files/mywav_reduced_noise" + str(m) + "filtered.wav", 'r')
    number_of_frames = w.getnframes()
    frames = w.readframes(number_of_frames)

    # Mozilla DeepSpeech model perform speech to text
    data16 = np.frombuffer(frames, dtype=np.int16)
    text = model.stt(data16)
    converted_texts.append(text)
    w.close()
    m += 1

if language == "en":
    transcript_texts = ["Where is the check-in desk?", "Where is the check-in desk?",
                        "I have lost my parents.", "I have lost my parents.", 
                        "Please, I have lost my suitcase.", "Please, I have lost my suitcase.", 
                        "What time is my plane?", "What time is my plane?", 
                        "Where are the restaurants and shops?","Where are the restaurants and shops?",
                        "Where is the departure hall?", "How do I get to the arrival hall?"]
if language == "es":   
    transcript_texts = ["¿Dónde están los mostradores?", "He perdido a mis padres.", 
                        "Por favor, he perdido mi maleta.", "¿A qué hora es mi avión?", 
                        "¿Dónde están los restaurantes y las tiendas?"] 
if language == "it":
    transcript_texts = ["Dove e' il bancone?", "Ho perso i miei genitori.", 
                        "Per favore, ho perso la mia valigia.", "A che ora e' il mio aereo?", 
                        "Dove sono i ristoranti e i negozi?"]
    

for i, text in enumerate(transcript_texts):
    # remove punctuations and convert everything to lower case
    final_text = "".join(character for character in text if character not in ("?", ".", ";", ":", "!", ",", "¿", "'"))
    final_text = final_text.replace("-", " ")
    final_text = final_text.lower()
    transcript_texts[i] = final_text

sum_of_SDI = 0
N = 0
total_WER = 0
total_SDI = 0
total_N = 0
for i, text in enumerate(converted_texts):
    print("Recognised text: " + text)
    print("Transcript text: " + transcript_texts[i])
    text_list = text.split(" ")
    transcript_text_list = transcript_texts[i].split(" ")
    
    # any substitution and deletion errors will be detected as all those scenarios will result in unmatched texts
    # correct number of texts in transcript text minus the total number of matches gives us the number of errors found
    substitution_and_deletion_errors = len(transcript_text_list) - len(set(text_list).intersection(transcript_text_list))
    
    # insertion errors can be calculated by subtracting length of text by length of transcript text
    # abs() is applied as deletion error occurs when length of text_list is lower than transcript while the opposite 
    # represents insertion error
    length_diff = len(transcript_text_list) - len(text_list)
    if length_diff < 0:
        insertion_errors = abs(length_diff)
    else:
        insertion_errors = 0
    sum_of_SDI = substitution_and_deletion_errors + insertion_errors
    N = len(transcript_text_list)
    print("Number of errors: " + str(sum_of_SDI))
    print("Total number of words: " + str(N))
    current_WER = sum_of_SDI/N * 100
    print("WER for current transcript: " + str(current_WER) + "%")
    total_SDI += sum_of_SDI
    total_N += N

total_WER = total_SDI/total_N * 100
print("Overall " + language + " WER: " + str(total_WER) + "%")

# Substitutions – When the system transcribes one word in place of another. Transcribing the fifth word as this instead of the is an example of a substitution error.
# Deletions – When the system misses a word entirely. In the example, the system deleted the first word well.
# Insertions – When the system adds a word into the transcript that the speaker didn’t say, such as or inserted at the end of the example.
# *) Word error rate (WER) can be computed as:
# WER = (S + D + I)/N * 100
# where
# • 𝑆 is the number of substitutions,
# • 𝐷 is the number of deletions,
# • 𝐼 is the number of insertions,
# • 𝑁 is the number of words in the sentence