# =[Modules dan Packages]========================
# =[Modules dan Packages]========================

from flask import Flask, render_template, request
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization

app = Flask(__name__)
model = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/", methods=['GET', 'POST'])
def score_comment():
    if request.method == 'POST':
        comment = request.form.get("comment")  # Mendapatkan nilai 'comment' dari formulir POST
        vectorized_comment = vectorizer([comment])
        results = model.predict(vectorized_comment)

        scores = {}
        for idx, col in enumerate(df.columns[2:]):
            scores[col] = results[0][idx] > 0.5

        return render_template('index.html', scores=scores, comment=comment)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    # Load model yang telah ditraining
    model = tf.keras.models.load_model('toxicity.h5')

    # Load dataset
    df = pd.read_csv(os.path.join('preprocessed_indonesian_toxic_tweet.csv'))

    # Inisialisasi vectorizer
    MAX_FEATURES = 200000
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                                   output_sequence_length=1800,
                                   output_mode='int')
    vectorizer.adapt(df['Tweet'].values)

    # Run Flask di localhost
    app.run(host="localhost", port=5000, debug=True)
