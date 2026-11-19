# 🚀 Data Autopilot AI

[![CI](https://github.com/arda-can/data-autopilot-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/arda-can/data-autopilot-ai/actions/workflows/ci.yml)

Bir CSV veya Excel dosyası verin; veriyi temizlesin, en anlamlı grafikleri seçsin, bir model eğitsin ve sorularınızı Türkçe yanıtlasın. Her adım deterministik, kayıt altında ve KVKK'ya duyarlı.

## Neyi farklı yapıyor

- **LLM kod çalıştırmaz, yalnızca plan yapar.** Gemini, Pydantic şemasıyla doğrulanan bir JSON planı döndürür; dönüşümleri önceden yazılmış güvenli fonksiyonlar uygular. Kod tabanında `exec`/`eval` yoktur.
- **Kişisel veri LLM'e gitmez.** Ham satır gönderilmez. E-posta, telefon, TC kimlik, IBAN ve kart numarası maskelenir; hassas sütunların örnekleri ve istatistikleri tümüyle gizlenir.
- **Veri sızıntısı yok.** Eksik doldurma ve ölçekleme yalnızca eğitim bölmesinden öğrenilir. Temizlenmiş veride eksikler ve aykırı değerler uydurulmaz, yalnızca raporlanır.
- **Kalite iki ayrı ölçü.** Yapısal hijyen skoru biçimi (boşluk, tekrar, tip) ölçer; iş kuralı uyumu ise tanımladığınız kurallara (aralık, benzersiz, boş olamaz, izinli değerler, desen, iki sütun karşılaştırması) uyan kayıtların oranıdır. Kurallar veriden önerilir ya da elle eklenir, JSON olarak kaydedilir; ihlal eden kayıtlar indirilebilir.
- **Müşteri analizi.** RFM segmentleri (önerilen aksiyonlarla), kohort tutundurma ısı haritası ve edinimden sonraki 3/6/12 ayda müşteri başına tarihsel gelir; tanımlar ve zaman penceresi ekranda yazılı.
- **Neden değişti?** Bir metriğin (sayım, toplam, ortalama, oran) iki dönem arasındaki değişimi boyut değerlerinin katkılarına tam olarak ayrıştırılır; ortalamalarda karma etkisi ile grup içi değişim ayrı gösterilir. Demo veride Q4 2025'teki puan düşüşünün kaynağı Ankara olarak bulunur.
- **A/B testi.** Oran (z-testi) ya da ortalama (Welch) karşılaştırması, fark ve göreli artış için güven aralığı, örneklem oranı uyumu (SRM) alarmı, yakalanabilir en küçük fark ve gereken örneklem. Atama müşteri bazındaysa analiz müşteri düzeyinde yapılır.
- **Her değişiklik kayıtlı.** Hangi sütuna ne yapıldığı ve kaç satırın etkilendiği audit log'da görünür.
- **Modeller bir taban çizgisiyle kıyaslanır.** Hiçbir özelliğe bakmayan taban modeli geçemeyen sonuç, "tahmin edilemiyor" olarak açıkça bildirilir.
- **Zaman serisi otopilotu.** Tarih sütunu bulununca trend, haftalık/yıllık döngü, olağan dışı dönemler ve %80 aralıklı tahmin; model, naif tahmine karşı geriye dönük test edilir ve geçemezse bunu söyler.
- **Dönem karşılaştırması.** "2017 ile 2018 arasında ne değişti?" sorusuna şema, eksik oranı ve dağılım farkıyla trafik ışığı tablosu.
- **Çoklu tablo.** `data/` içindeki tabloların ortak anahtarları bulunur; satır sayısını koruyan, onaylı bir birleştirme planıyla tek tabloya getirilir.
- **Şeffaf sohbet.** "İstanbul için 2018'de aylık harcama" gibi koşullu sorular; her cevabın altında eşdeğer pandas ve SQL.
- **Açıklanabilir model.** Kısmi bağımlılık grafikleri, Türkçe yorum ve what-if paneli; olası hedef sızıntısı uyarısı.
- **Tek tıkla yönetici raporu.** Özet, öneriler, grafikler ve model sonucu HTML (etkileşimli) veya PDF olarak indirilir.
- **API anahtarı olmadan da çalışır.** Anahtar yoksa kurallı yerel planlayıcı devreye girer.

## Kurulum

Python 3.13 ile geliştirildi (en az 3.11 gerekir).

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Gemini API anahtarı isteğe bağlıdır ve arayüzdeki kenar çubuğundan girilir.

## Veri

Depoda iki sentetik veri seti var; ikisi de gerçek kişi verisi içermez:

- **`data/demo_eticaret_siparisleri.csv`** (kenar çubuğunda "Demo e-ticaret verisini yükle"): 2024–2025 arası 7.057 sipariş, 4.000 müşteri. `core/demo_data.py` sabit tohumla üretir ve hangi örüntülerin bilerek konduğunu belgeler: tekrar eden müşteriler (RFM ve kohort için), gecikmeye, kategoriye ve indirime bağlı iade ve puan (öğrenilebilir sinyal), 2025'in ikinci yarısında mobil kanala kayma, Ekim 2025'ten itibaren Ankara'da teslimat gecikmesi, Temmuz 2025'te başlayan müşteri bazlı bir A/B deneyi (öneri kutusu) ve sayısı kayıtlı kural ihlalleri (negatif tutar, 1'den büyük indirim oranı, siparişten önce teslim, 1–5 dışı puan, geçersiz e-posta, şehir yazım farkı, tekrarlanan sipariş numarası, mükerrer satır). Ad, e-posta ve telefon sütunları KVKK maskesini göstermek içindir.
- **`data/kirli_musteri_verisi.csv`** ("Örnek Kirli Türkçe Veri Üret ve Yükle"): biçim temizliğini gösteren 200 satırlık küçük örnek.

Daha büyük bir deneme için [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) veri setini indirip CSV dosyalarını `data/` klasörüne koyun; uygulama onları otomatik listeler. Bu dosyalar boyutları nedeniyle depoya dahil değildir.

## Testler

```bash
pip install -r requirements-dev.txt
pytest
```

Testler `tests/fixtures/` altındaki sentetik veriyle çalışır; Olist dosyalarına ihtiyaç duymaz.

Testler API anahtarı kullanmaz: kurallı yerel planlayıcı, maskeleme ve LLM'e gidecek istem metni test edilir, ancak Gemini'ye yapılan gerçek çağrı test kapsamında değildir.

## Mimari

```
app.py               Streamlit giriş noktası: veri kaynağı ve sayfa gezintisi
views/               Sayfalar: Genel Bakış, Kalite, Zaman Serisi, Model, Müşteri, Deney, Rapor (yalnızca sunum)
core/ingestion.py    Dosya okuma, encoding ve ayraç tespiti
core/columns.py      Sütun tipi çıkarımı: metin, sayı, tarih, kimlik
core/health.py       Yapısal hijyen skoru ve aykırı değer raporu
core/rules.py        Veri kuralları ve iş kuralı uyumu
core/customers.py    RFM, kohort tutundurma, tarihsel müşteri değeri
core/rootcause.py    "Neden değişti?" katkı ayrıştırması
core/experiment.py   A/B testi, SRM, güç ve örneklem
core/metadata.py     PII maskeli metadata kartı
core/planner.py      Gemini planlayıcı + kurallı fallback
core/executor.py     Deterministik dönüşümler ve audit log
core/autopilot.py    Orkestrasyon, grafik seçimi, hedef önerisi
core/automl.py       Sızıntısız model karşılaştırması
core/chat.py         Doğal dil sorusu → yapılandırılmış sorgu (+ pandas/SQL karşılığı)
core/timeseries.py   Zaman serisi: trend, mevsimsellik, anomali, tahmin
core/drift.py        Dönem karşılaştırması: şema, eksik oranı, PSI
core/relations.py    Çoklu tablo: anahtar keşfi ve güvenli birleştirme
core/explain.py      Kısmi bağımlılık ve what-if
core/report.py       Yönetici raporu: HTML ve PDF
```

Ayrıntılı analiz, bilinen sorunlar ve yol haritası: [ANALIZ.md](ANALIZ.md).
