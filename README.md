# 🚀 Data Autopilot AI

[![CI](https://github.com/arda-can/data-autopilot-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/arda-can/data-autopilot-ai/actions/workflows/ci.yml)

Bir CSV veya Excel dosyası verin; veriyi temizlesin, en anlamlı grafikleri seçsin, bir model eğitsin ve sorularınızı Türkçe yanıtlasın. Her adım deterministik, kayıt altında ve KVKK'ya duyarlı.

## Neyi farklı yapıyor

- **LLM kod çalıştırmaz, yalnızca plan yapar.** Gemini, Pydantic şemasıyla doğrulanan bir JSON planı döndürür; dönüşümleri önceden yazılmış güvenli fonksiyonlar uygular. Kod tabanında `exec`/`eval` yoktur.
- **Kişisel veri LLM'e gitmez.** Ham satır gönderilmez. E-posta, telefon, TC kimlik, IBAN ve kart numarası maskelenir; hassas sütunların örnekleri ve istatistikleri tümüyle gizlenir.
- **Veri sızıntısı yok.** Eksik doldurma ve ölçekleme yalnızca eğitim bölmesinden öğrenilir. Temizlenmiş veride eksikler ve aykırı değerler uydurulmaz, yalnızca raporlanır.
- **Her değişiklik kayıtlı.** Hangi sütuna ne yapıldığı ve kaç satırın etkilendiği audit log'da görünür.
- **Modeller bir taban çizgisiyle kıyaslanır.** Hiçbir özelliğe bakmayan taban modeli geçemeyen sonuç, "tahmin edilemiyor" olarak açıkça bildirilir.
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

Depoda yalnızca küçük bir örnek var: `data/kirli_musteri_verisi.csv`. Arayüzdeki "Örnek Kirli Türkçe Veri Üret ve Yükle" düğmesi de aynı dosyayı üretir.

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
app.py               Streamlit arayüzü (yalnızca sunum)
core/ingestion.py    Dosya okuma, encoding ve ayraç tespiti
core/columns.py      Sütun tipi çıkarımı: metin, sayı, tarih, kimlik
core/health.py       Kalite skoru ve aykırı değer raporu
core/metadata.py     PII maskeli metadata kartı
core/planner.py      Gemini planlayıcı + kurallı fallback
core/executor.py     Deterministik dönüşümler ve audit log
core/autopilot.py    Orkestrasyon, grafik seçimi, hedef önerisi
core/automl.py       Sızıntısız model karşılaştırması
core/chat.py         Doğal dil sorusu → yapılandırılmış sorgu
core/report.py       Yönetici raporu: HTML ve PDF
```

Ayrıntılı analiz, bilinen sorunlar ve yol haritası: [ANALIZ.md](ANALIZ.md).
