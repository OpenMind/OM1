# OpenMind OM1 - Docker Kurulum Rehberi (macOS)

## ✅ Kurulum Tamamlandı

OpenMind OM1 sistemi başarıyla Docker ile kuruldu ve çalışıyor!

## 📋 Kurulum Özeti

### 1. Native Kurulum (UV ile)
- ✅ UV package manager kuruldu
- ✅ Sistem bağımlılıkları kuruldu (portaudio, ffmpeg)
- ✅ OM1 repository klonlandı
- ✅ Python 3.10 sanal ortamı oluşturuldu
- ✅ Tüm Python bağımlılıkları yüklendi
- ✅ Sistem başarıyla çalıştırıldı: `uv run src/run.py spot`

### 2. Docker Kurulum
- ✅ Docker image başarıyla oluşturuldu
- ✅ macOS için özel yapılandırma hazırlandı
- ✅ Container başarıyla çalıştırıldı
- ✅ WebSim arayüzü aktif: http://localhost:8000

## 🚀 Kullanım

### Native Çalıştırma (UV)
```bash
cd OM1
uv run src/run.py spot
```

### Docker ile Çalıştırma
```bash
cd OM1

# Container'ı başlat
docker-compose -f docker-compose.mac.yml up om1

# Arka planda çalıştırmak için
docker-compose -f docker-compose.mac.yml up -d om1

# Logları görüntüle
docker-compose -f docker-compose.mac.yml logs -f om1

# Container'ı durdur
docker-compose -f docker-compose.mac.yml down
```

## 🔑 API Key Ayarlama

Şu anda placeholder bir API key kullanılıyor. Gerçek API key almak için:

1. https://portal.openmind.org/ adresine git
2. Ücretsiz API key al
3. `.env` dosyasını düzenle:
```bash
nano OM1/.env
```

4. `OM_API_KEY` değerini güncelle:
```
OM_API_KEY=your_real_api_key_here
```

5. Sistemi yeniden başlat

## 🌐 Web Arayüzü

WebSim arayüzüne tarayıcıdan erişebilirsiniz:
- URL: http://localhost:8000
- Spot robot simülatörü ve kontrol paneli

## 📁 Dosya Yapısı

```
OM1/
├── docker-compose.yml          # Orijinal Docker Compose (Linux için)
├── docker-compose.mac.yml      # macOS için Docker Compose
├── Dockerfile                  # Orijinal Dockerfile (Linux için)
├── Dockerfile.mac              # macOS için Dockerfile
├── .env                        # Ortam değişkenleri (API key)
├── config/                     # Yapılandırma dosyaları
│   └── spot.json5             # Spot robot yapılandırması
└── src/
    └── run.py                 # Ana çalıştırma dosyası
```

## 🔧 Teknik Detaylar

### Disk Kullanımı
- Native kurulum: ~5-8 GB
- Docker image: ~2-3 GB
- Toplam: ~10-15 GB

### Sistem Gereksinimleri
- macOS (test edildi)
- Docker Desktop 27.4.1+
- Docker Compose v2.31.0+
- Python 3.10 (UV tarafından otomatik yönetilir)

### macOS Özel Ayarlamalar
- PulseAudio kontrolleri devre dışı bırakıldı
- Ses sistemi gereksinimleri kaldırıldı
- Network mode: host (localhost:8000 erişimi için)

## 🐛 Sorun Giderme

### API Key Hatası (401 Unauthorized)
```
ERROR - OpenAI API error: Error code: 401 - {'error': 'malformed API key'}
```
**Çözüm:** `.env` dosyasında gerçek API key kullanın.

### Container Başlatma Hatası
```bash
# Container'ı yeniden oluştur
docker-compose -f docker-compose.mac.yml down
docker-compose -f docker-compose.mac.yml build --no-cache om1
docker-compose -f docker-compose.mac.yml up om1
```

### Port 8000 Kullanımda
```bash
# Port'u kullanan process'i bul
lsof -i :8000

# Process'i durdur
kill -9 <PID>
```

## 📚 Ek Kaynaklar

- Resmi Dokümantasyon: https://docs.openmind.org/
- GitHub Repository: https://github.com/OpenMind/OM1
- API Portal: https://portal.openmind.org/

## ✨ Özellikler

- 🤖 Spot robot simülatörü
- 🎤 Ses tanıma (ASR)
- 🗣️ Konuşma sentezi (TTS)
- 👁️ Görüntü işleme (VLM)
- 🧠 LLM entegrasyonu (OpenAI uyumlu)
- 🌐 Web tabanlı kontrol paneli
- 📡 ROS2 ve Zenoh desteği
- 🔒 Asimov yasaları ile yönetişim

## 🎯 Sonraki Adımlar

### 1. Portal Kayıt ve Badge Alma (ÖNCELİKLİ!)
- �️ **portal.openmind.org** adresine git
- 🎖️ Google ile kayıt ol → Badge al
- 🎖️ WorldCoin doğrulama (opsiyonel) → Badge al
- 🎖️ Backpack wallet bağla (opsiyonel) → Badge al
- 💰 Airdrop puanları kazan

### 2. API Key Al
- ✅ https://portal.openmind.org/ adresinden gerçek API key al
- ✅ `.env` dosyasını güncelle
- ✅ Container'ı yeniden başlat

### 3. GitHub Contribution (Opsiyonel)
- 🔱 Repository'yi fork et
- ⭐ Star ver
- 📝 "help wanted" issue'lara bak
- 🤝 Telegram developer grubuna katıl: https://t.me/openminddev

### 4. Sistem Keşfi
- 🌐 http://localhost:8000 adresinden arayüzü keşfet
- 📖 Dokümantasyonu oku
- 🤖 Farklı robot yapılandırmalarını dene
- 🔧 Kendi özel modlarını oluştur

---

**⚠️ ÖNEMLİ:** Badge almak için **portal.openmind.org**'a kayıt olman gerekiyor! GitHub fork yapmak badge kazandırmıyor. Detaylı bilgi için `OPENMIND_BADGE_REHBERI.md` dosyasını oku.

**Not:** Bu kurulum macOS için optimize edilmiştir. Linux sistemlerde orijinal `docker-compose.yml` ve `Dockerfile` kullanılmalıdır.
