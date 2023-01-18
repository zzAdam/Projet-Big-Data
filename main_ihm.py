import speech_recognition as sr
import tkinter as tk
from tkinter import ttk
import requests

text = ""

def trad_en_to_ro(text_to_trad):


	url = "https://api.mymemory.translated.net/get"
	payload = {'q': text_to_trad, 'langpair': 'en|ro'}

	response = requests.get(url, params=payload)
	data = response.json()
	translated_text = data['responseData']['translatedText']
	return translated_text

def on_press():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        label.config(text=text)
        label2.config(text=trad_en_to_ro(text))
    except:
        label.config(text="Could not understand audio")

root = tk.Tk()
root.title("Push-to-Talk Speech Recognition")

style = ttk.Style()
style.configure("TButton", font=("Arial", 18))

title_label = ttk.Label(root, text="Speech Recognition", font=("Arial", 18, "bold"))
title_label.pack()

button = ttk.Button(root, text="Speak", command=on_press, width=10)
button.pack(pady=20)

label_origin = ttk.Label(root, text="Original Text:", font=("Arial", 16, "bold"))
label_origin.pack()
label = ttk.Label(root, text="", font=("Arial", 16))
label.pack()
label_trad = ttk.Label(root, text="Translated Text:", font=("Arial", 16, "bold"))
label_trad.pack()
label2 = ttk.Label(root, text="", font=("Arial", 16))
label2.pack()
root.mainloop()
