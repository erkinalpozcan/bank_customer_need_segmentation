### Uygulama akışı
1. bert.py ile bert model, önceden hazırlanmış veriler ile (csv dosyası) eğitilir
2. Modeli eğitirken confidence skoru arttıracağı için türk bir bert model tercih ettik, confidence skoru arttıracağı için
3. bert_test.py ile eğitilen model (my_modell.pt) bir arayüz ile çalıştırılır.
4. Ekrandaki entry kısmına, kategorisi bulunmak istenen banka sorgusu girilir ve gönder'e tıklanır.
5. Output olarak, hangi kategoride olduğu görülür. 

*Eğer model skoru görülmek istenirse, bert_test.py dosyasından label2.config fonskiyonuna (93.satır) {predicted_score} eklenebilir
