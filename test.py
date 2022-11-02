import speech_recognition as sr
import webbrowser as web

def main():
    safari_path = "/Applications/Safari.app"
    web.register('Safari', None,web.BackgroundBrowser(safari_path))
    web.register
    r = sr.Recognizer()


    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Please say something ")
        audio = r.listen(source)
        print("Reconizing Now")
        try:
            dest=r.recognize_google(audio)
            print("You have said: " + dest)
            web.get('Safari').open(dest)
        except Exception as e:
            print("Error:" + str(e))

if __name__=="__main__":
    main()
