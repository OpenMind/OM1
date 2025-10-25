<div align="center">
<img src="https://raw.githubusercontent.com/OpenMind/OM1/main/docs/assets/om1_logo.png" alt="OM1 Logo" width="500"/>
</div>

# OM1: Robotlar için Modüler Yapay Zeka Çalışma Zamanı

**[Website](https://openmind.org/)** | **[Dokümantasyon](https://docs.openmind.org/)** | **[Discord](https://discord.gg/openmindagi)**

OM1, geliştiricilerin dijital ortamlar ve fiziksel robotlar (İnsansılar, Telefon Uygulamaları, web siteleri, Dört Ayaklılar ve TurtleBot 4 gibi eğitim robotları dahil) arasında çok modlu yapay zeka ajanları oluşturmasını ve dağıtmasını sağlayan modüler bir yapay zeka çalışma zamanıdır (runtime).

OM1 ajanları, web verileri, sosyal medya, kamera yayınları ve LIDAR gibi çeşitli girdileri işleyebilirken; hareket, otonom navigasyon ve doğal konuşmalar gibi fiziksel eylemleri de mümkün kılar. OM1'in amacı, farklı fiziksel form faktörlerine uyum sağlamak için yükseltilmesi ve yeniden yapılandırılması kolay, son derece yetenekli, insan odaklı robotlar yaratmayı kolaylaştırmaktır.

## 🚀 Başlarken

### Ön Koşullar

- **Python 3.10 veya 3.11**
- **Poetry:** Paket yönetimi ve bağımlılıklar için.
- **Docker:** (Sadece Mac veya Linux için) Belirli modüllerin (örn. ses işleme) çalıştırılması için gereklidir.

### Kurulum

Projeyi klonlayın ve bağımlılıkları kurun:

```bash
git clone [https://github.com/OpenMind/OM1.git](https://github.com/OpenMind/OM1.git)
cd OM1
poetry install
poetry shell
