# Data Autopilot AI

Güvenli ve deterministik veri hijyeni + AutoEDA/AutoML platformu. Streamlit, pandas, Pydantic, scikit-learn ve Google GenAI SDK (`google-genai`). Portföy projesi; demoda dürüst ve etkileyici görünmesi önemli.

## Komutlar

Windows, sanal ortam `.venv` içinde:

- Tüm testler: `.venv\Scripts\python.exe -m pytest`
- Tek modül: `.venv\Scripts\python.exe -m pytest tests/test_executor.py`
- Uygulama: `.venv\Scripts\python.exe -m streamlit run app.py`

Python 3.13 kullanılıyor (pandas 3 en az 3.11 ister; `.cursorrules`'taki "3.10+" artık geçerli değil). Sürümler `requirements.txt` içinde sabit, geliştirme araçları `requirements-dev.txt` içinde. Yeni bağımlılık eklemeden önce kullanıcıya sor, sonra sürümüyle birlikte ekle. `python-docx` için ön onay var (henüz kurulu değil); `reportlab`, `kaleido`, `matplotlib` ve `statsmodels` kurulu ve sabit. `prophet` kullanılmayacak, `shap` yerine önce PDP.

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
- `app.py` (giriş, veri kaynağı, gezinti) ve `views/` (sayfalar) yalnızca sunum; iş mantığı `core/` içinde.
- Sayfalar arası durum `views/common.py` üzerinden: Streamlit çizilmeyen widget'ın durumunu sildiği için sayfa dışında da gereken seçimler `remember`/`remembered` ile saklanır ve widget'a varsayılan olarak verilir (widget anahtarına Session State ile değer yazma). Yeni veri yüklenince `set_dataset` hepsini unutur. Rapor sayfası diğer sayfaların son seçimlerini bu kayıttan okur.
- Kullanıcıya görünen tüm metinler, açıklamalar ve loglar **Türkçe**.

## Analizden çıkan kurallar

Bunların her biri gerçek bir hatadan doğdu (ayrıntı: `ANALIZ.md`).

- **Sütun tipi çıkarımı yalnızca `core/columns.py` üzerinden.** `dtype == "object"` yazma: pandas 3'te metin dtype'ı `str`. `is_text_series` / `text_columns` kullan.
- **Sütun adında alt dize arama yok.** `"id" in name` yerine `is_identifier_name` (kelime eşleşmesi); yoksa `width`, `humidity`, `paid` kimlik sayılır.
- **Sayıya çevirme tam eşleşmeyle.** Rakam dışını silip sayıya çevirme; `parse_numeric_text` kullan. Hash kimlikler, adresler ve baştaki sıfırlı kodlar metin kalmalı.
- **Tarihte kör `dayfirst=True` yok.** ISO tarihler gün/ay takasına uğrar; `parse_datetime_text` kullan.
- **Hedef sütuna ve eksik değerlere dokunma.** Aykırı değerler yalnızca raporlanır (`OutlierFinding`).
- **Kalite skoru veya metrik parlatmak için veri uydurma.** Demo dürüst olmalı.
- **Demo veri (`core/demo_data.py`) üreticiyle aynı kalır.** Üreticiyi değiştirirsen `write_demo_data()` ile `data/demo_eticaret_siparisleri.csv`'yi yeniden yaz; test ikisini karşılaştırır. Yeni bir özelliğin örüntüsü gerekiyorsa üreticiye ekle ve `DEMO_STORY`/`injected` içinde belgele.
- **Her hata düzeltmesi bir regresyon testiyle gelir.** Testler `tests/fixtures/` altındaki sentetik veriyi kullanır; Olist dosyaları depoda yok.

## Bilinen açıklar

Güncel liste ANALIZ.md bölüm 3–4'te. Kod yazarken akılda tutulacaklar:

- `AuditLog` alanı `affected_count`; `.cursorrules` `rows_affected` diyor.
- Gemini yolu gerçek anahtarla test edilmedi; testler sahte SDK yanıtı kullanır (`tests/test_gemini_paths.py`).
- Sızıntı uyarısı zamansal sırayı bilemez; kullanıcı "tahmin anı" ayarlarıyla (bilinmeyen sütunu çıkar / bilindiğini onayla) düzeltir. Sütun otomatik çıkarılmaz.
- Leaderboard metrik adları (`R2 Score`, `Accuracy`) hâlâ İngilizce (D1).

## Doğruluk kuralları (son inceleme turundan)

- **Okuma:** ayraç `csv` modülüyle tırnak kurallarına göre seçilir; baştaki sıfırlı sütunlar metin okunur; okuma hataları Türkçe `ValueError` olur ve arayüz mesajla gösterir. `.xls` desteklenmez (xlrd yok).
- **Veri koruma:** LLM planı dolu ve değişken bir sütunu silemez, değerlerin %90'ından azı sayıysa dönüştüremez; reddedilen öneri `RejectedAction` olarak kaydedilir.
- **Önbellek:** anahtarlar içerikten türetilir (`content_signature`, `frame_signature`); bir sonucu etkileyen her ayar (hedef, tahmin anı, zaman sütunu) anahtara girer.
- **LLM:** yanıt `request_structured` ile alınır (`model_validate_json`); hata yutulmaz, `PlanOrigin` kaynağı ve nedeni taşır.
- **AutoML:** model yalnızca eğitim bölmesinde çapraz doğrulamayla seçilir; test bölmesi seçime girmez. Rapor eğitim/test satırlarını ve örneklemi belirtir.
- **Audit:** örnekler değişen satırlardan seçilir, `sample_rows` kaynağı gösterir.
- **Uçtan uca testler:** `tests/test_app_e2e.py`, `AUTOPILOT_DATA_DIR` ile sentetik tablolar kullanır; yeni arayüz akışı eklenirse buraya test eklenir.

## Yol haritası

`ANALIZ.md` bölüm 4 (güncel); ilk analiz Ek A'da. Tamamlananlar: kritik düzeltmeler, AutoML taban modeli ve sızıntı uyarısı, HTML/PDF yönetici raporu (`core/report.py`), zaman serisi (`core/timeseries.py`), dönem karşılaştırması (`core/drift.py`), çoklu tablo (`core/relations.py`), sohbet filtreleri ve SQL şeffaflığı (`core/chat.py`), PDP ve what-if (`core/explain.py`). Yol haritasındaki beş özellik bitti; sıradaki adaylar ANALIZ.md 5.0'da.

Rapor kuralları: `ExecutiveReport` JSON'a çevrilebilir kalmalı; ham satır ve audit örnekleri rapora girmez; veriden gelen her metin HTML'de `html.escape`, PDF'te `xml_escape` ile kaçırılır.

PDF grafik akışı (`core/chart_export.py`): `render_pdf(report, figures)` en fazla altı görsel basar (özellik önemi çubuğu dahil). Her Plotly figürü açık temalı bir kopyaya çevrilir ve önce Kaleido ile PNG'ye aktarılır (Kaleido 1.x Chrome ister). Kaleido başarısız olursa hata yutulmaz: `core.chart_export` logger'ına uyarı düşer, Matplotlib (GUI'siz Agg) yedeği çizer ve PDF'e "yedek çizici" notu eklenir. 3B ve paralel koordinat grafikleri PDF'e aktarılmaz; yalnızca HTML'de bulunur. Yeni grafik tipi eklenirse `_matplotlib_png` içinde karşılığı yazılmalı, yoksa yedek yol `ValueError` verir.

Sayı ve grafik sunumu `core/formatting.py` üzerinden Türkçe biçimlenir (`decimal_tr`, `integer_tr`, `display_frame`, `localize_figure`); bunlar yalnızca sunum kopyası üretir, veriyi değiştirmez. `find_category_variants` 5.000'den fazla benzersiz değerli sütunları taramaz.

Zaman serisi kuralları (`core/timeseries.py`):

- Frekans otomatik seçilir: aylık → haftalık → günlük sırasıyla, mevsimselliği (iki tam döngü + tahmin ufku) ölçülebilen ve dönemlerinin en fazla %20'si boş olan ilk frekans. Hiçbiri yoksa mevsimsellik olmadan en kaba yoğun frekans.
- Kenar dönemler sessizce silinmez ve güncel veri atılmaz: seyrek dönemler (tipik değerin %10'u altı) seriyi ancak uzun bir boşlukla (günlükte en az 14 gün) bölümlere ayırır; en az 12 dönemlik **en güncel** bölüm seçilir, daha kısa boşluklar 0 sayılıp not edilir. Tam kapsanmayan ilk/son hafta-ay kendi takvim uzunluğuyla ölçülür (Şubat 28 gün). Sağdan kesilme kontrolü son dönemi aynı mevsimsel fazdaki önceki dönemlerle kıyaslar (cumartesi önceki cumartesilerle); olağan hafta sonu düşüşü kesim sayılmaz. Atılan her şey `notes` içinde yazılır.
- Model seçimi her eğitim penceresinde yalnızca eğitim verisiyle AICc üzerinden yapılır (sönümlü / doğrusal / trendsiz ETS); geriye dönük test penceresine bakılmaz. Model mevsimsel naif tahmini %5 geçemezse hüküm bunu açıkça söyler; tahmin yine gösterilir.
- Mevsimsellik gücü olağan dışı dönemler (|z| ≥ 4) hariç ölçülür; tek bir kampanya günü haftalık döngüyü gizlememeli.
- statsmodels 0.15'te `ETSModel` numpy girdiyle `get_prediction` çağrısında çöküyor; modele tamsayı indeksli `pd.Series` verilir.
- Tarih eksenlerinde Plotly'nin İngilizce ay adları yerine `tickformat="%m.%Y"` kullanılır.

AutoML hedef sızıntısı (`AutoMLEngine._leakage_check`): model tabanı geçtiğinde iki imza aranır. Eksiklik imzası: bir sütunun yalnızca boş/dolu olması kazancın en az %50'sini açıklıyor (sonuç gerçekleştikten sonra doldurulan alan, ör. Olist `order_delivered_customer_date` → `order_status`). Baskınlık imzası: en önemli beş özellikten biri tek başına kazancın en az %90'ını sağlıyor. Sütun otomatik çıkarılmaz; uyarı arayüzde, özette ve önerilerde gösterilir ve şüphe varken özellikler "öncelikli izlenmeli" diye önerilmez.

Dönem karşılaştırması (`core/drift.py`): trafik ışığı etki büyüklüğüne göre (PSI 0,10 / 0,25; eksik oranı 5 / 10 puan). p-değeri büyük örneklemde her farkı anlamlı gösterdiği için yalnızca bilgidir. Kimlik ve serbest metin sütunlarında dağılım karşılaştırılmaz, doluluk değişimi raporlanır. Sayısal PSI'da referans aralığının altı ve üstü ayrı dilimdir (sabit referansta kayma PSI = 0 görünmesin). İkinci dosya `prepare_for_comparison` ile aynı yapısal hijyenden geçer.

Çoklu tablo (`core/relations.py`): yalnızca iki birleştirme türü vardır; "eşleştir" (karşı anahtar benzersiz, `validate="many_to_one"`) ve "topla" (anahtar başına sayım/ortalama/toplam/en sık). Çoktan-çoğa ilişki önerilmez. Birleştirme satır sayısını asla değiştirmemeli; her adımın eşleşme oranı kaydedilir. Boş anahtar hiçbir kayıtla eşleşmez (pandas NA'yı NA ile eşleştirir; sağdaki boş anahtarlar önceden çıkarılır). Toplamada yalnızca adı kimlik/kod olan sütunlar atlanır; tam sayı olmak kimlik göstergesi değildir.

Sohbet (`core/chat.py`): filtreler `QueryFilter` ile tipleri doğrulanarak maskeyle uygulanır; `eval`/`DataFrame.query` kullanılmaz. `explain()` eşdeğer pandas ve PostgreSQL kodunu yalnızca gösterim için üretir; bu metinler asla çalıştırılmaz. Anahtar kelime ve kategori değeri ararken kelime sınırı kullan ("toplam" içinde "top", "teslimat" içinde "teslim" geçer); uzun eşleşme önce gelir. Gruplama yalnızca soruda ipucu varsa ("göre", "bazında", "başına") ya da çok değerli filtrede yapılır. Filtrenin kullandığı sütun adı hedef aranırken sayılmaz, ama ad yeniden geçiyorsa aynı sütun hedef olabilir.

Veri kuralları (`core/rules.py`): tek düz şema `DataRule` (tür + isteğe bağlı parametreler); türler aralık, benzersiz, boş olamaz, izinli değerler, desen, karşılaştırma. İşleçler `operator` modülüne eşlenir; `eval`/`query` yok (test kaynağı tarar). Desen en fazla 200 karakter, iç içe niceleyici reddedilir, değerin ilk 256 karakterine uygulanır. Boş değer yalnızca "boş olamaz" kuralını ihlal eder. Saatsiz bir tarih saatli bir tarihle gün düzeyinde kıyaslanır. Uyum skoru satır düzeyindedir (hiçbir kuralı ihlal etmeyen kayıtların oranı); kural yoksa skor `None`, "100" uydurulmaz. Uygulanamayan kural sessizce atlanmaz, hata metniyle raporlanır. Öneriler (`suggest_rules`) gerekçe taşır ve onay kutusuyla kapatılabilir; Gemini önerileri şema + `check_rule` doğrulamasından geçer, reddedilenler gösterilir. Rapora yalnızca kural başına sayılar girer, ihlal eden değerler girmez. Eski "Kalite Skoru" arayüzde ve raporda "Yapısal Hijyen" adını taşır.

Müşteri analizi (`core/customers.py`): tanımlar `DEFINITIONS` içinde yazılıdır ve arayüzde/raporda aynen gösterilir. İşlem = müşteri, tarih ve pozitif tutarı dolu kayıt (sipariş numarası varsa sipariş başına toplanır); dışarıda kalanlar sayılıp not edilir. Analiz tarihi = son işlem günü + 1. RFM skorları 1–5 yüzdelik dilim, eşit değerler aynı skoru alır (`rank(method="min")`); segment R ve F skorundan, her R-F çifti tam bir segmente eşlenir. Pencere yalnızca RFM'i sınırlar; kohortlar tüm veriden. Gözlenemeyen kohort hücreleri NaN'dır, 0 değil. CLV tarihsel ve gelir bazlıdır: edinimden sonraki N ayı tamamen gözlenmiş müşterilerin ortalaması; tahmin modeli yok. Rapora müşteri kimliği girmez.

Kök neden (`core/rootcause.py`): katkıların toplamı değişime birebir eşit olmalı (test her boyutta doğrular). Toplam/sayımda katkı = grup farkı; ortalama/oranda simetrik shift-share: karma etkisi Δw·(m̄ − M̄) + grup içi değişim w̄·Δm. Bir dönemde olmayan grubun grup içi değişimi 0'dır. Boyutlar "beklenenden sapma" ile sıralanır (değişim gruplara eşit yayılsaydı beklenen katkıya uzaklık), büyük grup yalnızca büyük olduğu için öne çıkmaz. Kimlik/PII/50'den fazla değerli sütun ve metrik sütunu boyut olmaz; doluluğu dönemler arasında 30 puandan fazla değişen boyut (yeni toplanan alan) atlanır ve nedeni yazılır. Nedensellik iddiası yok; yöntem notu her sonuçta gösterilir. Dönem bölme `core/drift.py` (`split_by_ranges`, `split_by_years`, `split_by_date`).

Açıklanabilirlik (`core/explain.py`): model kara kutu (yalnızca `predict`/`predict_proba`), arka plan test bölmesinden en fazla 500 satır. Sızıntı şüphesi olan özellikler gizlenmez, uyarıyla gösterilir. AutoML tarih çiftleri için "sonraki − önceki (gün)" özelliği üretir (satır içi hesap, sızıntısız).
