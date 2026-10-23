# General File RAG MCP — Unsloth Desktop

Bu sunucu, `workspace` klasörüne koyduğun belgeleri metne dönüştürür, MiniLM embedding'leri oluşturur ve kalıcı SQLite indeksinde saklar. Arama **hibrit** çalışır: vektör benzerliği + SQLite FTS5 BM25, RRF ile birleştirilir; bölüm başlığıyla eşleşen sonuçlar ek puan alır. Böylece hem "vergi indirimi" gibi konu sorguları hem de "IX. The Frontier" gibi bölüm adı sorguları doğru parçaları getirir.

## Desteklenen formatlar
- PDF (metin katmanı olan PDF)
- TXT, Markdown, CSV/TSV, JSON/JSONL, XML, HTML, RTF, LOG, YAML, TOML
- Kod dosyaları: Python, JS/TS, CSS, SQL, PowerShell, shell, C/C++, Java, Go, Rust vb.
- DOCX, XLSX, PPTX
- PNG, JPG/JPEG, TIFF, BMP, WEBP (OCR)

**OCR notu:** Görseller için ayrıca Windows'a Tesseract OCR kurulmalı; `tesseract.exe` PATH üzerinde olmalı ve `tur`/`eng` dil verileri yüklenmeli. Varsayılan OCR dili `tur+eng`. Taranmış PDF sayfaları otomatik OCR'lanmaz; PDF'de metin katmanı yoksa metin çıkarılamayabilir.

## Nasıl çalışır
- Metin, başlık-farkında bölümlere ayrılır (`CHAPTER IX`, `# Başlık`, `Bölüm 3`, `Section 1` vb.).
- Her bölüm, varsayılan 550 karakterlik (100 karakter örtüşmeli) chunk'lara bölünür.
- Chunk embedding'i, bölüm başlığı öne eklenerek üretilir; başlık ayrıca veritabanında saklanır ve aramada kullanılır.
- Arama sırasında vektör sıralaması ile BM25 sıralaması RRF ile birleştirilir; sorgu kelimeleri başlıkta geçiyorsa hibrit skor yükseltilir.
- Her sonuç `path`, `chunk`, `heading`, `similarity`, `hybrid_score` ve `text` alanlarını döndürür.
- FTS5 derleme desteği yoksa arama otomatik olarak yalnızca vektör moduna düşer ve yanıtta `search_mode` bunu belirtir.

## Kurulum
1. ZIP'i `C:\Users\mstf\Documents\general-rag-mcp` konumuna çıkar.
2. Bu klasörde PowerShell aç ve bağımlılıkları kur:
   ```powershell
   uv sync
   ```
3. Belgelerini `workspace` klasörüne kopyala.
4. `.env` dosyasını açıp `RAG_WORKSPACE`, `RAG_DB_PATH` ve isteğe bağlı diğer ayarları kontrol et (aşağıdaki [Ortam değişkenleri](#ortam-değişkenleri-env) bölümü).
5. Unsloth MCP ayarına `.mcp.json` içeriğini ekle; yolun doğru olduğundan emin ol.
6. Unsloth'ı yeniden başlat veya MCP sunucusunu yeniden bağla; ilk indekslemede embedding modeli indirilir.

`.mcp.json` artık ortam değişkeni içermez; ayarlar proje kökündeki `.env` dosyasından okunur. Böylece yolu tek yerden değiştirmen yeterlidir.

### `.mcp.json` örneği
```json
{
  "mcpServers": {
    "general-rag": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\Users\\mstf\\Documents\\general-rag-mcp",
        "python",
        "server.py"
      ]
    }
  }
}
```

> **Şema notu:** Bu sürüm `heading` sütunu, FTS5 tablosu ve daha küçük chunk'lar kullanır. Önceki sürümden kalan indeks ilk açılışta otomatik silinir; belgeleri bir kez yeniden indekslemen gerekir. Bunu sunucu stderr'inde görürsün: "Şema güncellendi: eski indeks temizlendi..."

## Ortam değişkenleri (`.env`)

Sunucu, proje kökündeki `.env` dosyasını başlangıçta okur. Öncelik sırası:

1. Süreç ortam değişkenleri (Unsloth, terminal, `.mcp.json` içindeki `env`)
2. `.env` dosyasındaki değerler
3. Kod içindeki varsayılanlar

Yani `.env` değerleri mevcut ortam değişkenlerini **ezmez**. Değişiklikten sonra MCP sunucusunu yeniden başlat. Yorumlar `#` ile başlar; boşluk içeren yolları çift tırnak içine al.

| Değişken | Varsayılan | Açıklama | İndeks gerektirir mi? |
|---|---|---|---|
| `RAG_WORKSPACE` | `<proje>\workspace` | Sunucunun erişebildiği tek kök klasör. Mutlak yol önerilir; yoksa otomatik oluşturulur. | Evet, yeni konumda yeniden indeksle |
| `RAG_DB_PATH` | `<proje>\rag_index.sqlite3` | Chunk'ların, embedding'lerin ve FTS tablosunun tutulduğu SQLite dosyası. | Evet, yeni DB boş başlar |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face model kimliği veya yerel model yolu. Çok dilli, 384 boyutlu, CPU dostu. | Evet, `force=true` ile tüm dosyalar |
| `RAG_CHUNK_SIZE` | `550` | Chunk başına azami karakter. Küçük değer daha isabetli ama daha çok chunk; büyük değer daha az chunk ama seyrelmiş benzerlik. | Evet, `force=true` |
| `RAG_CHUNK_OVERLAP` | `100` | Ardışık chunk'lar arası örtüşme. `CHUNK_SIZE`'ın %40'ını geçmemeli. | Evet, `force=true` |
| `RAG_MAX_FILE_MB` | `100` | Tek dosya için MB cinsinden üst sınır; aşan dosya indekslenmez. | Hayır |
| `RAG_RRF_K` | `60` | Vektör + BM25 sıralamalarını birleştiren RRF sabiti. Küçük değer üst sıralara daha çok ağırlık verir. | Hayır |
| `RAG_HEADING_BOOST` | `0.45` | Sorgu kelimeleri başlıkta geçtiğinde uygulanan çarpan. `0` kapatır; `1`+ agresiftir. | Hayır |
| `RAG_OCR_LANG` | `tur+eng` | Görsel OCR için Tesseract dil(ler)i. Kurulu dil paketleriyle eşleşmeli. | OCR'lanan dosyalar için evet |
| `RAG_ENV_FILE` | `<proje>\.env` | Alternatif env dosyası yolu. Süreç ortamında tanımlanmalıdır; `.env` içinde tanımlanırsa dikkate alınmaz. | Hayır |

Ayrıntılı açıklamalar ve örnekler `.env` dosyasının içinde yorum olarak bulunur.

### Yaygın ayar senaryoları
- **Daha isabetli arama:** `RAG_CHUNK_SIZE=450`, `RAG_CHUNK_OVERLAP=120`, ardından tüm dosyaları `force=true` ile yeniden indeksle.
- **Daha hızlı / küçük indeks:** `RAG_CHUNK_SIZE=900`, `RAG_CHUNK_OVERLAP=80`.
- **Bölüm adı sorguları zayıf:** `RAG_HEADING_BOOST` değerini `0.7`'ye çıkar (indeks gerekmez).
- **Başlık bonusu sonuçları bozuyor:** `RAG_HEADING_BOOST=0` yap.
- **Çok büyük belgeler:** `RAG_MAX_FILE_MB` değerini yükselt; RAM ve işlem süresini göz önünde bulundur.
- **Farklı dil ağırlıklı belgeler:** Daha güçlü çok dilli bir model (örn. `BAAI/bge-m3`) seçip tüm indeksi yeniden oluştur.

## MCP araçları
- `list_files`: workspace'teki desteklenen dosyaları listeler.
- `index_file(path, force=false)`: tek dosyayı indeksler. Dosya değişmediyse `already_indexed` döner; `force=true` yeniden indeksler.
- `index_directory(path=".", recursive=true, force=false)`: klasördeki dosyaları indeksler; hatalı dosyalar sonuç listesinde ayrıca raporlanır.
- `search_documents(query, limit=5, path_filter="", expand_neighbors=false)`: hibrit arama. `path_filter` ile dosya/klasör daraltılır; `expand_neighbors=true` her sonuca önceki/sonraki chunk bağlamını ekler.
- `list_sections(path="")`: indeksteki bölüm/başlıkları chunk aralıklarıyla listeler. Bölüm adını bulmak için kullan.
- `read_section(path, heading="", max_chars=30000)`: bir bölümün tüm chunk'larını sırayla birleştirip döndürür; `heading` boşsa dosyanın tamamını verir. Başlık eşleşmesi büyük/küçük harf ve noktalama farklarını tolere eder ("IX. The Frontier" → "CHAPTER IX — THE FRONTIER"). Bulunamazsa mevcut başlıkları listeler.
- `read_file(path, max_chars=12000)`: dosyadan metin çıkarır; indeksleme yapmaz.
- `remove_from_index(path)`: yalnızca indeks kayıtlarını siler; asıl dosyaya dokunmaz.

### Önerilen kullanım akışı
1. `index_directory(path=".", recursive=true)` ile belgeleri indeksle.
2. `list_sections` ile bölüm adlarını gör.
3. Konu veya bölüm adıyla `search_documents` çağır.
4. Tam metin gerekiyorsa `read_section` veya `read_file` kullan.
5. Sonuçlardaki `path` ve `chunk` alanlarını kaynak olarak göster.

## Unsloth'ta örnek komutlar
- `index_directory` aracını `path="."`, `recursive=true` ile çağır ve belgeleri indeksle.
- `search_documents` ile "IX. The Frontier" ara; `heading` alanı eşleşen sonuçları göster.
- `list_sections` ile `rapor.pdf` içindeki bölümleri listele.
- `read_section` ile `rapor.pdf` dosyasının "IX. The Frontier" bölümünün tam metnini getir.
- `search_documents` ile "vergi indirimi" konusunu ara, `expand_neighbors=true` kullan ve en alakalı 8 sonucu getir.
- `search_documents` çağrısına `path_filter="raporlar/"` ekleyerek yalnızca bir alt klasörde ara.

## Güvenlik ve sınırlar
- Sunucu yalnızca `RAG_WORKSPACE` altındaki dosyalara erişir. `../` gibi kaçış denemeleri reddedilir; dışarıdaki dosyaları önce workspace'e kopyala.
- Varsayılan maksimum dosya boyutu 100 MB'dir (`RAG_MAX_FILE_MB`).
- Dosyalar otomatik indekslenmez; `index_file` veya `index_directory` çağrısı gerekir.
- Embedding'ler yerel olarak hesaplanır; belge içeriği bir API'ye gönderilmez. Model ilk kez indirilirken Hugging Face'e bağlanılır.
- Anlamsal arama sonuçları kesin hüküm değildir; kaynak metni `read_section`/`read_file` ile doğrula.

## Sorun giderme
- **İlk indeksleme çok yavaş:** Embedding modeli indiriliyor ve CPU'da çalışıyor olabilir. İlk çağrı normalden uzundur; sonrakiler hızlanır.
- **"İndeks boş" hatası:** Önce `index_file` veya `index_directory` çalıştır.
- **Arama başlıkla bulmuyor:** `list_sections` ile başlığın indekste olduğunu doğrula; `RAG_HEADING_BOOST` değerini artır ve sunucuyu yeniden başlat.
- **Ayar değişikliği etkisiz:** Değişkenin süreç ortamında da tanımlı olup olmadığını kontrol et (ortam değişkenleri `.env`'i ezer) ve sunucuyu yeniden başlat.
- **Türkçe karakterler bozuk görünüyor:** Yalnızca PowerShell konsol kod sayfası kaynaklı olabilir; MCP yanıtları UTF-8 JSON'dur. Kalıcı olarak düzeltmek için PowerShell'de `chcp 65001` kullan.
- **OCR çalışmıyor:** `tesseract --list-langs` ile dil paketlerini, `tesseract --version` ile PATH'i kontrol et.
- **FTS5 yok:** Bazı Python/sqlite derlemelerinde FTS5 kapalıdır. Sunucu otomatik olarak vektör-only moda geçer; `search_mode` alanından görülür.
