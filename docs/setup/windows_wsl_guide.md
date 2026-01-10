# OM1 Windows Kurulumu (WSL ile)

Bu rehber, Windows kullanan ve Linux bilmeyen kişiler için hazırlanmıştır.

## 1. WSL Nedir?
WSL, Windows içinde Linux çalıştırmanızı sağlar. OM1 Windows’ta doğrudan çalışmadığı için gereklidir.

## 2. WSL Kurulumu
PowerShell’i yönetici olarak açın ve şu komutu yazın:

wsl --install

Bilgisayar yeniden başlayacaktır.

## 3. Ubuntu Açma
Başlat menüsünden "Ubuntu" yazın ve açın.
Kullanıcı adı ve şifre oluşturun.

## 4. Docker Kurulumu
Docker Desktop indirin:
https://www.docker.com/products/docker-desktop/

Kurulumdan sonra:
- “Use WSL 2 backend” açık olmalı
- Ubuntu entegrasyonu aktif olmalı

## 5. OM1 Kurulumu
Ubuntu terminalinde:

git clone https://github.com/OpenMind/OM1.git
cd OM1
pip install -e .

## 6. Mikrofon Sorunu
WSL’de mikrofon doğrudan çalışmayabilir.
Geçici çözüm olarak metin tabanlı input kullanılabilir.

## 7. Test
OM1’in çalıştığını görmek için:

om1 run spot
