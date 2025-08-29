from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report

app = Flask(__name__)

# Model ve vectorizer'ı global olarak tanımla
vectorizer = None
model = None

def train_model():
    """Spam tespit modelini eğit ve kaydet"""
    global vectorizer, model
    
    # Veri seti (50 e-posta örneği)
    emails = [
        # SPAM örnekleri (25)
        "ÖNEMLİ!!! Hemen tıkla 1000$ kazan!!!",
        "BEDAVA iPHONE kazan! Şimdi tıkla!",
        "Kredi kartı bilgilerinizi güncelleyin HEMEN!",
        "Zengin olmanın sırrı burada! 5000TL kazanın!",
        "BÜYÜK İNDİRİM!!! %90 indirim kaçırma!!!",
        "Şanslı gününüz! Hemen para kazanın!",
        "Ücretsiz deneme! Şimdi başvur!",
        "Süper teklif! Sadece bugün geçerli!",
        "Hemen kaydol, bedava ödüller kazan!",
        "Büyük fırsat! Kazanç garantili!",
        "Kazanç fırsatı: şimdi başvur, hemen kazan!",
        "Hediye kartı kazanmak için tıklayın",
        "Sana özel teklif: bedava hediye kartı!",
        "Sadece bugün: bonus puan kazan!",
        "Hemen tıklayın ve büyük ödülü kapın",
        "Para kazanmak için tıklayın!",
        "Bedava bonus fırsatı! Kaçırma!",
        "Süper kazanç: hemen başvur!",
        "Hemen şimdi kazanç fırsatını yakala!",
        "Ödüller seni bekliyor, tıkla!",
        "Gizemli ödül kazanın şimdi!",
        "Şanslı gün: büyük ikramiye!",
        "Tıkla ve bedava hediyeni al!",
        "Sadece bugün: özel kazanma fırsatı!",
        "Hemen başvur ve ödülünü kap!"
    ]

    emails += [
        # NORMAL örnekleri (25)
        "Yarınki toplantı saat 14:00'da konferans salonunda",
        "Doğum günü partine gelir misin? Cumartesi saat 8'de",
        "Proje raporu ektedir, inceleyip geri dönüş yapabilirsin",
        "Market alışverişi yapacağım, sana da bir şey lazım mı?",
        "Sinema biletleri aldım, film saat 20:30'da başlıyor",
        "Bugün spor salonuna gideceğim",
        "Ödevimi bitirdim, gönderebilirim",
        "Araba servise gidecek, akşam eve geç kalabilirim",
        "Yeni kitap aldım, birlikte okuyalım mı?",
        "Hava bugün çok güzel, yürüyüşe çıkalım",
        "Toplantı saat 10'da, lütfen hazır olun",
        "Fatura ödemenizi geciktirmeyin, uyarı mesajı",
        "Kahve içmeye gelir misin?",
        "Spor salonu üyeliğini yenilemeyi unutma",
        "Yarın hava yağmurlu olacak, şemsiye al",
        "Arkadaşlarla sinemaya gideceğiz, gelmek ister misin?",
        "Webinar kaydı açıldı, katılımınızı bekliyoruz",
        "Ders notlarını paylaşabilir misin?",
        "Akşam yemeği için ne hazırlayalım?",
        "Yarınki ders için notları hazırla",
        "Kitap kulübü toplantısı saat 19:00'da",
        "Ödev teslim tarihini unutma",
        "Saat 15:00'te randevuya gel",
        "Bugün hava güzel, parkta buluşalım",
        "Toplantı saat 11:30'da başlayacak"
    ]

    labels = [1]*25 + [0]*25  # SPAM=1, NORMAL=0

    # DataFrame
    df = pd.DataFrame({"text": emails, "label": labels})

    # Eğitim ve test seti
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=labels)

    # TF-IDF Vektörleştirme
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Ensemble Model
    log_model = LogisticRegression(max_iter=200)
    nb_model = MultinomialNB()

    ensemble = VotingClassifier(estimators=[('lr', log_model), ('nb', nb_model)], voting='soft')
    ensemble.fit(X_train_tfidf, y_train)

    # Model performansı
    y_pred = ensemble.predict(X_test_tfidf)
    
    # Model ve vectorizer'ı kaydet
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(ensemble, "spam_model.pkl")
    
    # Global değişkenlere ata
    vectorizer = vectorizer
    model = ensemble
    
    return {
        'accuracy': (y_pred == y_test).mean(),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred)
    }

def load_model():
    """Kaydedilen modeli yükle"""
    global vectorizer, model
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        model = joblib.load("spam_model.pkl")
        return True
    except:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """Modeli eğit"""
    try:
        results = train_model()
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    """E-posta analiz et"""
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text:
            return jsonify({'success': False, 'error': 'E-posta metni boş olamaz'})
        
        # Model yüklü değilse yükle
        if vectorizer is None or model is None:
            if not load_model():
                return jsonify({'success': False, 'error': 'Model yüklenemedi. Önce eğitim yapın.'})
        
        # TF-IDF ile dönüştür
        features = vectorizer.transform([email_text])
        
        # Tahmin ve olasılık
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        
        result = {
            'success': True,
            'prediction': 'SPAM' if pred == 1 else 'NORMAL',
            'spam_probability': round(prob[1] * 100, 1),
            'normal_probability': round(prob[0] * 100, 1),
            'confidence': round(max(prob) * 100, 1)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Uygulama başladığında modeli yüklemeye çalış
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
