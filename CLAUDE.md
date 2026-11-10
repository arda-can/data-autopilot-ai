# Data Autopilot AI

Güvenli ve deterministik veri hijyeni + AutoEDA/AutoML platformu. Streamlit, pandas, Pydantic, scikit-learn ve Google GenAI SDK (`google-genai`). Portföy projesi; demoda dürüst ve etkileyici görünmesi önemli.

## Komutlar

Windows, sanal ortam `.venv` içinde:

- Tüm testler: `.venv\Scripts\python.exe -m pytest`
- Tek modül: `.venv\Scripts\python.exe -m pytest tests/test_executor.py`
- Uygulama: `.venv\Scripts\python.exe -m streamlit run app.py`

Python 3.13 kullanılıyor (pandas 3 en az 3.11 ister; `.cursorrules`'taki "3.10+" artık geçerli değil). Sürümler `requirements.txt` içinde sabit, geliştirme araçları `requirements-dev.txt` içinde. Yeni bağımlılık eklemeden önce kullanıcıya sor, sonra sürümüyle birlikte ekle. `statsmodels` ve `python-docx` için ön onay var (henüz kurulu değil); `reportlab`, `kaleido` ve `matplotlib` kurulu ve sabit. `prophet` kullanılmayacak, `shap` yerine önce PDP.

Makinede Smart App Control açık: derlenmiş bir modül (`.pyd`) ilk yüklemede "Uygulama Denetimi ilkesi bu dosyayı engelledi" hatası verebilir. Aynı komutu bir kez daha çalıştırmak genellikle yeter.

## Mimari kurallar (katı)

1. **Ana hatta `exec()`/`eval()` yok.** LLM yalnızca planlayıcıdır; Pydantic ile doğrulanan JSON şeması döndürür.
2. **Deterministik aksiyon dağıtıcısı (`core/executor.py`).** Tüm dönüşümler önceden yazılmış, güvenli fonksiyonlardır ve aksiyon adlarına eşlenir.
3. **Önce KVKK/PII (`core/metadata.py`).** Ham satırlar LLM'e gitmez. LLM'e giden her metin (metadata kartı, sütun adı, kullanıcı sorusu) `MetadataGenerator` maskesinden geçer.
4. **Sıfır veri sızıntısı.** Eksik doldurma ve aykırı değer kırpma ön işlemede yapılmaz; yalnızca modelleme hattında, eğitim bölmesinden öğrenilir. Ön işleme yalnızca yapısal hijyendir (biçim düzeltme, boş/sabit sütun, mükerrer).
5. **Audit kaydı.** DataFrame'deki her değişiklik `AuditLog` olarak kaydedilir.
6. **Yerel `data/` desteği.** Arayüz hem `data/` içinden seçim hem dosya yükleme sunar.

## Kodlama standartları

- Tip ipuçları ve şemalar için `pydantic.BaseModel`.
- Girdi DataFrame'i asla yerinde değiştirme; `.copy()` ile başla.
- `app.py` yalnızca sunum; iş mantığı `core/` içinde.
- Kullanıcıya görünen tüm metinler, açıklamalar ve loglar **Türkçe**.

## Analizden çıkan kurallar

Bunların her biri gerçek bir hatadan doğdu (ayrıntı: `ANALIZ.md`).

- **Sütun tipi çıkarımı yalnızca `core/columns.py` üzerinden.** `dtype == "object"` yazma: pandas 3'te metin dtype'ı `str`. `is_text_series` / `text_columns` kullan.
- **Sütun adında alt dize arama yok.** `"id" in name` yerine `is_identifier_name` (kelime eşleşmesi); yoksa `width`, `humidity`, `paid` kimlik sayılır.
- **Sayıya çevirme tam eşleşmeyle.** Rakam dışını silip sayıya çevirme; `parse_numeric_text` kullan. Hash kimlikler, adresler ve baştaki sıfırlı kodlar metin kalmalı.
- **Tarihte kör `dayfirst=True` yok.** ISO tarihler gün/ay takasına uğrar; `parse_datetime_text` kullan.
- **Hedef sütuna ve eksik değerlere dokunma.** Aykırı değerler yalnızca raporlanır (`OutlierFinding`).
- **Kalite skoru veya metrik parlatmak için veri uydurma.** Demo dürüst olmalı.
- **Her hata düzeltmesi bir regresyon testiyle gelir.** Testler `tests/fixtures/` altındaki sentetik veriyi kullanır; Olist dosyaları depoda yok.

## Bilinen açıklar

- `AuditLog` alanı `affected_count`; `.cursorrules` `rows_affected` diyor. Örnekler etkilenen satırlardan değil ilk 5 satırdan alınıyor (ANALIZ.md O2).
- LLM planı için korkuluk ve hata görünürlüğü eksik; Gemini yolu gerçek anahtarla test edilmedi (ANALIZ.md Y4).

## Yol haritası

`ANALIZ.md` bölüm 5. Tamamlananlar: kritik düzeltmeler, AutoML taban modeli, HTML/PDF yönetici raporu (`core/report.py`). Sıradaki iş: zaman serisi otopilotu (`statsmodels`).

Rapor kuralları: `ExecutiveReport` JSON'a çevrilebilir kalmalı; ham satır ve audit örnekleri rapora girmez; veriden gelen her metin HTML'de `html.escape`, PDF'te `xml_escape` ile kaçırılır.

PDF grafik akışı (`core/chart_export.py`): `render_pdf(report, figures)` en fazla altı görsel basar (özellik önemi çubuğu dahil). Her Plotly figürü açık temalı bir kopyaya çevrilir ve önce Kaleido ile PNG'ye aktarılır (Kaleido 1.x Chrome ister). Kaleido başarısız olursa hata yutulmaz: `core.chart_export` logger'ına uyarı düşer, Matplotlib (GUI'siz Agg) yedeği çizer ve PDF'e "yedek çizici" notu eklenir. 3B ve paralel koordinat grafikleri PDF'e aktarılmaz; yalnızca HTML'de bulunur. Yeni grafik tipi eklenirse `_matplotlib_png` içinde karşılığı yazılmalı, yoksa yedek yol `ValueError` verir.

Sayı ve grafik sunumu `core/formatting.py` üzerinden Türkçe biçimlenir (`decimal_tr`, `integer_tr`, `display_frame`, `localize_figure`); bunlar yalnızca sunum kopyası üretir, veriyi değiştirmez. `find_category_variants` 5.000'den fazla benzersiz değerli sütunları taramaz.
