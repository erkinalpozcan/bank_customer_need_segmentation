from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F  
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from tkinter import font 


test_sentence = None


def kullan():
    global test_sentence
    test_sentence = entry.get()  
    model()


root = tk.Tk()
root.geometry("600x400")
root.title("Banka Müşteri Hizmetleri Sorgulama Uygulaması")


background_image = Image.open("background.jpg")  
background_image = background_image.resize((600, 400))  
bg_photo = ImageTk.PhotoImage(background_image)

canvas = tk.Canvas(root, width=600, height=400)
canvas.pack(fill="both", expand=True)  
canvas.create_image(0, 0, image=bg_photo, anchor="nw")  



baslık = tk.Label(root, text="Banka Müşteri Hizmetleri Sorgulama Uygulaması", font=( font.Font(family="Arial", size=12, weight="bold")),fg="blue")  
baslık_window = canvas.create_window(300, 100, anchor="center", window=baslık) 


canvas.create_text(300, 170, text="Karşılaştığınız problemi giriniz", font=("Arial", 10), fill="navyblue")


entry = tk.Entry(root, width=40)
entry_window = canvas.create_window(300, 200, anchor="center", window=entry)  


buton = tk.Button(root, text="Gönder", command=kullan, width=10, height=2, bg="blue", fg="white")
buton_window = canvas.create_window(305, 270, anchor="center", window=buton) 


label2 = tk.Label(root, text="", font=("Arial", 12), bg="lightgray")  
label2_window = canvas.create_window(300, 350, anchor="center", window=label2)  


# Model yükleme ve tahmin fonksiyonu
def model():
    model_save_path = r"C:\Users\Erkinalp\Desktop\piton\NLP\my_modell"  
    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    model = BertForSequenceClassification.from_pretrained(model_save_path)

    inputs = tokenizer(
        test_sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  

    probabilities = F.softmax(logits, dim=1)

    predicted_class = logits.argmax(dim=1).item()  
    predicted_score = probabilities[0][predicted_class].item()  

    
    label_map = {
        0: "Genel Sorular",
        1: "Güvenlik",
        2: "Hesap İşlemleri",
        3: "Kart İşlemleri",
        4: "Kredi İşlemleri",
        5: "Mobil Bankacılık",
        6: "Müşteri Hizmetleri",
        7: "Para Transferi",
        8: "Yatırım",
        9: "Şube İşlemleri"
    }
    predicted_label = label_map[predicted_class]
    
    
    label2.config(text=f"{predicted_label}") # Eğer confidence skor istiyorsan bunu da ekle   ,{predicted_score}


root.mainloop()
