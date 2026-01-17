![G20VrWxakAAGoEL](https://github.com/user-attachments/assets/a492888f-10f8-4af9-9b24-5fced20d5ba3)

# Deploy Your First Robot on OpenMind Fabric Portal (Linux)

Bu rehberde, Linux üzerinde OM1 agent kurulumunu adım adım öğrenip kurulum yapabileceksiniz..

---

## 1- Gerekli Paketlerin Kurulumu:

```bash
sudo apt update -y && sudo apt upgrade -y && sudo apt install -y portaudio19-dev python3-all-dev ffmpeg alsa-utils python3-pip && sudo pip install uv && sudo modprobe snd-dummy
```

---

## 2- UV Kurulumu:

```bash
wget https://astral.sh/uv/install.sh
chmod +x install.sh
./install.sh
```

---

## 3- OM1 Reposunu Klonla:

```bash
git clone https://github.com/openmind/OM1.git
cd OM1
```

---

## 4- UV, Git ve Sanal Ortam Kurulumu:

```bash
pip install uv
git submodule update --init
uv venv
```

### Örnek Çıktı:
<img width="1677" height="300" alt="image" src="https://github.com/user-attachments/assets/8b9affeb-e2e0-4856-b1de-4f0e05b84ba0" />

---


## 5- API Key Al

- [**OpenMind Fabric Portal**](https://fabric.openmindnetwork.xyz) bağlantısına git.
- OpenMind’e kayıt ol veya hesabın varsa giriş yap.
- Sağ üstteki menüden “Purchase Credits” kısmına tıkla ve Base ağı üzerinden 5 USDC bakiye ekle.
- Ardından “Create API Key” butonuna tıkla ve yeni bir API key oluştur.
- Oluşan anahtarı kopyalamayı unutma, çünkü pencereyi kapattıktan sonra tekrar göremezsin.

![1](https://github.com/user-attachments/assets/238fbf73-2bd8-432d-b324-9bdb992e6bbf)

---

## 6- API KEY Ekle (.env):

```bash
cp env.example .env
nano .env
```

- Dosyada "OM-API-KEY=" şu satırı bul:
- OpenMind üzerinden oluşturduğun kendi **API anahtarını** buraya ekle.
- Kaydedip çık:
> CTRL + X → Y → ENTER

![Adsız tasarım](https://github.com/user-attachments/assets/bff474bf-e617-451e-ad2b-1888245cb5cf)

---

## 6- Node’u Screen İçerisinde Başlat:

Arka planda kesintisiz çalışması için node’u `screen` içinde başlatıyoruz:

```bash
screen -S openmind
```

```bash
cd ~/OM1
source .venv/bin/activate
```
```bash
uv run src/run.py conversation
```

<img width="1679" height="497" alt="image" src="https://github.com/user-attachments/assets/c2318cbd-4292-44d8-88cd-b7e5f6549d87" />

- Artık agent terminalde çalışmaya başlayacak.

---

## 7- Screen Komutları:

Screen’den çıkmak için:
```bash
CTRL + A + D
```

Tekrar girmek için:
```bash
screen -r openmind
```

---

## 🔁 Yeniden Başlatma:

Sunucu yeniden başlarsa veya node kapanırsa:

```bash
screen -r openmind
```
```bash
cd ~/OM1
source .venv/bin/activate
uv run src/run.py conversation
```

---

## ⚠️ Sık Görülen Hatalar ve Çözümler:

| Hata                           | Sebep                            | Çözüm                                                                                                |
| ------------------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `uv: command not found`        | uv kurulu değil                  | `pip install uv` komutunu çalıştır.                                                                  |
| `.venv/bin/activate not found` | sanal ortam oluşturulmamış       | `uv venv` komutunu tekrar çalıştır.                                                                  |
| `401 Unauthorized`             | OpenMind bakiyesi yetersiz       | [https://portal.openmind.org](https://portal.openmind.org) adresine girip kredi ekle.                |
| `portaudio not found`          | ses modülü eksik                 | `sudo apt install portaudio19-dev -y`      komutunu çalıştır.                                        |
| `ERROR: No output from LLM`    | API key hatalı veya kredi bitmiş | `.env` dosyasındaki key’i kontrol et veya yenisini oluştur                                           |

---

### 💡 Ek İpuçları

* “401 Insufficient Balance” hatası görürsen eğer, hesabına kredi ekle ve node’u yeniden başlat.
* Node’u her zaman `screen` içinde çalıştır, böylece terminali kapatsan bile işlem arka planda devam eder.
