# Demo videosu çekim senaryosu

**Hedef süre:** 3 dakika (en fazla 3:30). **İzleyici:** veri analisti / veri bilimci işe alan yöneticiler ve teknik görüşmeciler. **Ana mesaj:** "Bu araç veriyi hızlıca analiz ediyor, ama asıl gösterdiği şey benim analizde neye dikkat ettiğim: dürüst metrik, sızıntı, doğru istatistik."

## Hazırlık

- Uygulamayı yerelde başlatın: `.venv\Scripts\python.exe -m streamlit run app.py`. Ya da yayınlandıysa Cloud bağlantısını kullanın.
- Tarayıcıda **`http://localhost:8501/?demo=1`** adresini bir kez açıp otopilotun bitmesini bekleyin; ikinci açılış önbellekten anında gelir. Kayıttan önce sayfayı yenileyin.
- Pencere: 1440×900 ya da 1920×1080, tarayıcı yakınlaştırması %100. Yer imleri çubuğunu ve bildirimleri kapatın.
- Kayıt: OBS ya da Windows Ekran Alıcısı (Win+Alt+R). Mikrofon ayrı kaydedilirse sonradan eşlemek kolaylaşır.
- Gemini anahtarı **girmeyin**; demo yerel planlayıcıyla çalışır ve anahtar ekranda görünmemeli.

## Akış

| Süre | Ekranda | Anlatım (öneri) |
|---|---|---|
| 0:00–0:15 | Genel Bakış, üstteki dört metrik | "Merhaba, ben Arda. Bu, Türkçe çalışan bir veri analizi otopilotu. Sentetik bir e-ticaret verisi yükledim: 7 bin sipariş, 4 bin müşteri. Verideki her örüntüyü ben koydum, bu yüzden aracın doğru bulup bulmadığını gösterebilirim." |
| 0:15–0:35 | Genel Bakış: tespit edilen sorunlar; aşağı kaydırıp grafiklere hızlıca göz atın | "Önce yapısal hijyen: mükerrer satırlar, sabit sütun, negatif tutar, şehir adı yazım farkları. Eksik değerleri doldurmuyor, aykırı değerleri kırpmıyor; bunlar yalnızca modelleme sırasında, eğitim verisinden öğrenilerek yapılıyor. Böylece veri sızıntısı olmuyor." |
| 0:35–1:00 | **Kalite** sayfası: iki skor, önerilen kurallar, kural tablosu; "İhlal eden kayıtları indir" düğmesini gösterin | "'Kalite 98' tek başına bir şey söylemez; değerlerin doğru olup olmadığını ölçmez. Bu yüzden ikinci bir skor var: iş kuralı uyumu. Kuralları veriden öneriyor: tutar negatif olamaz, teslim tarihi siparişten önce olamaz, puan 1–5 arası. Verideki bütün bilinçli hataları doğru sayıyla buluyor. Kurallar serbest kod değil; LLM de yalnızca bu şemayla kural önerebiliyor." |
| 1:00–1:35 | **Zaman Serisi** → aşağıda "Neden Değişti?": Metrik = Ortalama, Sütun = değerlendirme_puanı → "Değişimi ayrıştır"; şelale grafiği ve şehir tablosu | "Analistin en sık duyduğu soru: bu ay neden düştü? Son çeyrekte müşteri puanı 4,29'dan 4,06'ya inmiş. Araç değişimi boyutlara ayırıyor ve düşüşün tamamının Ankara'dan geldiğini buluyor. Ankara siparişlerin yüzde 15'i; değişim eşit yayılsaydı katkısı da o kadar olurdu. Önemli olan bu beklentiden sapma. Gerçekte de Ankara'da Ekim'den beri teslimat gecikmesi var. Ayrıca bunun nedensellik değil, ayrıştırma olduğunu açıkça yazıyor." |
| 1:35–2:00 | **Model** sayfası: sarı sızıntı uyarısı; sonra hedefi `iade_edildi` yapın | "Otopilot sipariş durumunu tahmin etmeyi denedi ve yüzde 99,6 F1 aldı. Ama bunun neden fazla iyi olduğunu da söylüyor: teslim tarihi sonradan doluyor, yani hedef sızıntısı. Bu yüzden güvenilir değil. İade tahminine geçince model taban çizgisini dürüstçe geçiyor ve sızıntı uyarısı yok." |
| 2:00–2:25 | **Müşteri** sayfası: RFM grafiği ve tablo, kohort ısı haritası | "Müşteri tarafında RFM segmentleri ve her segment için önerilen aksiyon var. 'Kaybedilmemeli' segmenti müşterilerin yüzde 6'sı ama gelirin yüzde 10'u. Kohort tablosunda henüz gözlenemeyen aylar boş bırakılıyor, sıfır sayılmıyor; aksi hâlde tutundurma olduğundan kötü görünürdü." |
| 2:25–2:55 | **Deney** sayfası: "Deneyi analiz et"; yeşil hüküm, SRM satırı, güç bölümü | "Son olarak A/B testi. Atama müşteri bazında yapılmış, bu yüzden analiz de müşteri düzeyinde; aynı müşterinin siparişleri bağımsız sayılmıyor. B grubunda ek ürün oranı yüzde 34 daha yüksek, p 0,007. Örneklem oranı uyumu kontrolü temiz. Sonuç anlamsız çıksaydı, kaç kullanıcı gerektiğini söyleyecekti." |
| 2:55–3:15 | **Rapor** sayfası: "Raporu Hazırla" → PDF'i açıp ilk sayfayı gösterin | "Bütün bunlar tek tıkla HTML ve PDF yönetici raporuna dönüşüyor. Rapora ham satır ya da müşteri kimliği girmiyor." |
| 3:15–3:30 | GitHub deposu: README ve yeşil CI rozeti | "Kod GitHub'da: 370'ten fazla test ve CI var. Mimari kural basit: LLM yalnızca plan yapar, hiçbir yerde eval yok, kişisel veri LLM'e gitmez. İzlediğiniz için teşekkürler." |

## Çekim ipuçları

- Her sayfada önce 1–2 saniye durun, sonra konuşun; izleyici ekranı okuyabilsin.
- İmleçle yalnızca konuştuğunuz yeri gösterin; hızlı kaydırmayın.
- Sayıları ekrandakiyle aynı söyleyin. Demo verisi sabit tohumla üretildiği için değerler her çalıştırmada aynıdır.
- LinkedIn için 60 saniyelik kısa kesit: 1:00–1:35 ("Neden değişti?") ile 2:25–2:55 (A/B testi) birleştirilebilir.
- Altyazı: Türkçe altyazı ekleyin; LinkedIn videoların çoğu sessiz izlenir.
