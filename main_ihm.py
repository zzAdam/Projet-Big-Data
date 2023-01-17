import speech_recognition as sr
import tkinter as tk
from tkinter import ttk

def on_press():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        label.config(text=text)
    except:
        label.config(text="Could not understand audio")

root = tk.Tk()
root.title("Push-to-Talk Speech Recognition")

style = ttk.Style()
style.configure("TButton", font=("Arial", 18))

button = ttk.Button(root, text="Speak", command=on_press, width=10)
button.pack(pady=20)

label = ttk.Label(root, text="", font=("Arial", 16))
label.pack()

root.mainloop()
