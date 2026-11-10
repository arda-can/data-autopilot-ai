# Data Autopilot AI - Copilot Talimatları

- Ana veri temizleme akışında kesinlikle `exec()` veya `eval()` kullanılmayacak.
- Tüm dönüşümler `core/executor.py` içindeki önceden tanımlı deterministik fonksiyonlarla yapılacak.
- KVKK/PII güvenliği için LLM'e veri giderken (v2'de) TC Kimlik, e-posta ve telefonlar maskelenecek.
- Yapılan her işlem AuditLog olarak kaydedilecek ve arayüz tamamen Türkçe olacak.
