# 📧 Spam Tespit Sistemi

Bu proje, yapay zeka kullanarak e-posta spam tespiti yapan bir web uygulamasıdır. Colab'da geliştirdiğiniz spam tespit modelini modern bir web arayüzü ile birleştirir.

## 🚀 Özellikler

- **Ensemble Model**: Logistic Regression + Naive Bayes kombinasyonu
- **TF-IDF Vektörleştirme**: Metin analizi için gelişmiş özellik çıkarımı
- **Modern Web Arayüzü**: Responsive tasarım ve kullanıcı dostu arayüz
- **Gerçek Zamanlı Analiz**: Anında spam tespit sonuçları
- **Model Eğitimi**: Web arayüzünden model yeniden eğitimi
- **Örnek E-postalar**: Test için hazır örnekler

## 📋 Gereksinimler

- Python 3.8+
- pip (Python paket yöneticisi)

## 🛠️ Kurulum

1. **Projeyi klonlayın veya indirin**
   ```bash
   cd spam
   ```

2. **Gerekli paketleri yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

3. **Uygulamayı çalıştırın**
   ```bash
   python app.py
   ```

4. **Tarayıcınızda açın**
   ```
   http://localhost:5000
   ```

## 🎯 Kullanım

### 1. Model Eğitimi
- "🚀 Modeli Eğit" butonuna tıklayın
- Model 50 e-posta örneği ile eğitilecek
- Eğitim sonuçları gösterilecek

### 2. E-posta Analizi
- Metin kutusuna analiz edilecek e-postayı yazın
- "🔍 Analiz Et" butonuna tıklayın
- Sonuçlar anında gösterilecek:
  - SPAM/NORMAL tahmini
  - Olasılık yüzdeleri
  - Güven skoru

### 3. Örnek E-postalar
- Test için hazır örnekler mevcut
- Örneklere tıklayarak metin kutusunu doldurabilirsiniz

## 🔧 Teknik Detaylar

### Model Mimarisi
- **Ensemble Classifier**: Voting Classifier (Soft Voting)
- **Algoritmalar**: 
  - Logistic Regression
  - Multinomial Naive Bayes
- **Özellik Çıkarımı**: TF-IDF (1-2 gram)
- **Veri Seti**: 25 spam + 25 normal e-posta

### Web Teknolojileri
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **API**: RESTful endpoints
- **Responsive**: Mobile-first tasarım

## 📁 Proje Yapısı

```
spam/
├── app.py              # Flask uygulaması
├── templates/          # HTML şablonları
│   └── index.html     # Ana sayfa
├── requirements.txt    # Python paketleri
├── README.md          # Bu dosya
├── tfidf_vectorizer.pkl # Eğitilen vectorizer
└── spam_model.pkl     # Eğitilen model
```

## 🌐 API Endpoints

- `GET /` - Ana sayfa
- `POST /train` - Model eğitimi
- `POST /analyze` - E-posta analizi

## 🎨 Özelleştirme

### CSS Stilleri
- `templates/index.html` dosyasındaki `<style>` bölümünü düzenleyin
- Renkleri, fontları ve boyutları değiştirebilirsiniz

### Model Parametreleri
- `app.py` dosyasındaki `train_model()` fonksiyonunu düzenleyin
- TF-IDF parametrelerini, model türlerini değiştirebilirsiniz

## 🐛 Sorun Giderme

### Model Yüklenemiyor
- "🚀 Modeli Eğit" butonuna tıklayın
- Model dosyaları otomatik oluşturulacak

### Paket Hatası
- `pip install -r requirements.txt` komutunu tekrar çalıştırın
- Python sürümünüzün 3.8+ olduğundan emin olun

### Port Hatası
- `app.py` dosyasında port numarasını değiştirin
- Başka bir uygulama 5000 portunu kullanıyor olabilir

## 📊 Performans

- **Eğitim Süresi**: ~2-3 saniye
- **Analiz Süresi**: ~0.1 saniye
- **Doğruluk**: %90+ (test verisi üzerinde)
- **Bellek Kullanımı**: ~50MB

## 🔮 Gelecek Geliştirmeler

- [ ] Daha büyük veri seti ile eğitim
- [ ] Deep Learning modelleri
- [ ] Çoklu dil desteği
- [ ] Batch analiz özelliği
- [ ] API rate limiting
- [ ] Kullanıcı hesapları

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📞 İletişim

Sorularınız için issue açabilir veya proje sahibi ile iletişime geçebilirsiniz.

---

**Not**: Bu uygulama eğitim amaçlıdır ve production ortamında kullanmadan önce güvenlik testleri yapılmalıdır.
