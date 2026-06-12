# 🧠 Google Colab で AI を育成するガイド

このガイドでは、Google Colab を使用して、「数理変換タクティクス」用の AI モデルを学習します。

---

## 📋 **ステップ1: Google Colab を開く**

1. [Google Colab](https://colab.research.google.com/) にアクセス
2. 「新しいノートブック」をクリック
3. ノートブック名を「Math-Tactics-AI-Training」に変更

---

## 🔧 **ステップ2: 環境セットアップ**

以下のコードをセル1に貼り付けて実行:

```python
# ライブラリのインストール
!pip install tensorflow keras numpy pandas matplotlib scikit-learn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

print("✅ ライブラリのインストール完了！")
print(f"TensorFlow バージョン: {tf.__version__}")
```

---

## 🎮 **ステップ3: ゲームデータ生成**

セル2に以下のコードを貼り付けて実行:

```python
# ゲーム状態のベクトル化関数
def generate_game_data(num_samples=5000):
    """
    ゲームの状態ベクトルと最適なアクション（ラベル）を生成
    
    状態ベクトル（30次元）:
    - [0-9]: 自分の手札10枚 (正規化)
    - [10]: 自分のスコア (正規化)
    - [11-20]: 敵の手札10枚 (正規化)
    - [21]: 敵のスコア (正規化)
    - [22]: ターン数 (正規化)
    - [23-29]: 予備
    
    アクション（3カテゴリ）:
    - 0: パス（何もしない）
    - 1: 攻撃（約数強奪）
    - 2: 合成（足し算または掛け算）
    """
    
    X = []  # 入力（状態ベクトル）
    y = []  # 出力（アクションラベル）
    
    for _ in range(num_samples):
        # ランダムな状態ベクトルを生成
        state = np.random.rand(30)
        
        # 自分のスコアと敵のスコアから最適なアクションを決定
        my_score = state[10]
        enemy_score = state[21]
        turn_count = state[22]
        
        # ゲームロジックベースの簡単なルール
        if my_score < 0.3 and turn_count > 0.2:
            # 遅れている場合は攻撃
            action = 1
        elif my_score > 0.7:
            # リードしている場合はパス
            action = 0
        else:
            # 中盤は合成で準備
            action = 2
        
        X.append(state)
        y.append(action)
    
    return np.array(X), np.array(y)

# データ生成
print("🎮 ゲームデータを生成中...")
X_train, y_train = generate_game_data(5000)

print(f"✅ データ生成完了！")
print(f"入力データ形状: {X_train.shape}")
print(f"出力データ形状: {y_train.shape}")
print(f"アクション分布: {np.bincount(y_train)}")
```

---

## 🧠 **ステップ4: AI モデル構築**

セル3に以下のコードを貼り付けて実行:

```python
# ニューラルネットワークモデルの構築
def build_model():
    """
    数理変換タクティクス用の AI モデルを構築
    """
    model = models.Sequential([
        # 入力層 (30次元)
        layers.Input(shape=(30,)),
        
        # 隠れ層1
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dropout(0.3),
        
        # 隠れ層2
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dropout(0.2),
        
        # 隠れ層3
        layers.Dense(32, activation='relu', name='hidden_3'),
        layers.Dropout(0.1),
        
        # 出力層 (3つのアクション)
        layers.Dense(3, activation='softmax', name='output')
    ])
    
    # モデルのコンパイル
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# モデル構築
print("🏗️ AI モデルを構築中...")
model = build_model()

# モデルアーキテクチャの表示
model.summary()
```

---

## 🚀 **ステップ5: AI をトレーニング**

セル4に以下のコードを貼り付けて実行:

```python
# モデルのトレーニング
print("📚 モデルをトレーニング中...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\n✅ トレーニング完了！")
```

---

## 📊 **ステップ6: 学習結果を可視化**

セル5に以下のコードを貼り付けて実行:

```python
# 学習曲線を表示
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 損失の推移
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# 精度の推移
axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Model Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"最終精度: {history.history['accuracy'][-1]:.4f}")
```

---

## 💾 **ステップ7: モデルを H5 形式で保存**

セル6に以下のコードを貼り付けて実行:

```python
# H5形式でモデルを保存
print("💾 モデルを H5 形式で保存中...")
model.save('math_tactics_model.h5')
print("✅ モデルを保存しました: math_tactics_model.h5")

# ファイルサイズを確認
file_size = os.path.getsize('math_tactics_model.h5') / (1024 * 1024)
print(f"ファイルサイズ: {file_size:.2f} MB")
```

---

## 🔄 **ステップ8: TensorFlow.js 形式に変換**

セル7に以下のコード���貼り付けて実行:

```python
# TensorFlow.js 形式に変換
print("🔄 TensorFlow.js 形式に変換中...")
!pip install tensorflowjs

import tensorflowjs as tfjs

# モデルを tfjs 形式で保存
tfjs.converters.save_keras_model(model, 'tfjs_model')
print("✅ TensorFlow.js 形式で保存しました: tfjs_model/")

# 生成されたファイルを確認
!ls -la tfjs_model/
```

---

## 📤 **ステップ9: GitHub Pages にアップロード**

### 手順1: ファイルをダウンロード

```python
# Colab からファイルをダウンロード
from google.colab import files

# H5 モデルをダウンロード
files.download('math_tactics_model.h5')

# tfjs_model フォルダをまとめてダウンロード
!zip -r tfjs_model.zip tfjs_model/
files.download('tfjs_model.zip')
```

### 手順2: GitHub リポジトリにアップロード

1. GitHub で `m24039-source/game-` リポジトリを開く
2. 「Add file」→「Upload files」をクリック
3. `tfjs_model/` フォルダ内のファイルをアップロード（`model.json` と `.bin` ファイル）
4. Commit & Push

### 手順3: GitHub Pages で公開

1. リポジトリの Settings → Pages
2. Source を「main」に設定
3. 「Save」をクリック

URL: `https://m24039-source.github.io/game-/tfjs_model/`

---

## 🧪 **ステップ10: モデルをテスト**

セル8に以下のコードを貼り付けて実行:

```python
# テスト用のランダムな状態ベクトルでモデルをテスト
test_state = np.random.rand(1, 30)
prediction = model.predict(test_state)

action_names = ['パス', '攻撃', '合成']
predicted_action = np.argmax(prediction[0])
confidence = prediction[0][predicted_action]

print(f"入力状態: {test_state[0][:5]}...（最初の5次元のみ表示）")
print(f"\n予測結果:")
for i, (name, prob) in enumerate(zip(action_names, prediction[0])):
    print(f"  {i}. {name}: {prob:.4f}")
print(f"\n最適なアクション: {action_names[predicted_action]} (信頼度: {confidence:.4f})")
```

---

## 🎯 **完全なコード（コピペ用）**

以下のコードをすべてコピーして、新しい Colab ノートブックの**1つのセル**に貼り付けて実行することもできます:

```python
# ========== セットアップ ==========
!pip install tensorflow keras numpy pandas matplotlib scikit-learn tensorflowjs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import tensorflowjs as tfjs

print("✅ ライブラリのインストール完了！")

# ========== データ生成 ==========
def generate_game_data(num_samples=5000):
    X = []
    y = []
    for _ in range(num_samples):
        state = np.random.rand(30)
        my_score = state[10]
        enemy_score = state[21]
        turn_count = state[22]
        
        if my_score < 0.3 and turn_count > 0.2:
            action = 1  # 攻撃
        elif my_score > 0.7:
            action = 0  # パス
        else:
            action = 2  # 合成
        
        X.append(state)
        y.append(action)
    
    return np.array(X), np.array(y)

print("🎮 ゲームデータを生成中...")
X_train, y_train = generate_game_data(5000)
print(f"✅ データ生成完了! 形状: {X_train.shape}")

# ========== モデル構築 ==========
print("🏗️ AI モデルを構築中...")
model = models.Sequential([
    layers.Input(shape=(30,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========== トレーニング ==========
print("📚 モデルをトレーニング中...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# ========== 結果の可視化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Model Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ========== 保存 ==========
print("💾 モデルを保存中...")
model.save('math_tactics_model.h5')
tfjs.converters.save_keras_model(model, 'tfjs_model')
print("✅ モデルを保存しました!")

# ========== ダウンロード ==========
from google.colab import files
files.download('math_tactics_model.h5')
!zip -r tfjs_model.zip tfjs_model/
files.download('tfjs_model.zip')
print("✅ ダウンロード準備完了!")
```

---

## 📚 **次のステップ**

1. ✅ **モデルの改善**
   - ゲームの実際のプレイデータで学習させる
   - より複雑なゲームロジックを実装する
   - ハイパーパラメータを調整する

2. 📤 **GitHub に自動アップロード**
   - GitHub Actions で自動デプロイ設定

3. 🎮 **ゲームでテスト**
   - ゲーム画面で「AIと戦う」をクリック
   - AI が新しいモデルを使用して思考

---

## 💡 **ヒント**

- **より強いAIにするには**: トレーニングデータを増やす（10000→50000）
- **高速化するには**: GPU を使用（Colab のランタイムを GPU に変更）
- **カスタマイズするには**: モデルアーキテクチャを変更（層を増やすなど）

---

**Happy AI Training! 🚀**
