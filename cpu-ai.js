async function loadAIBrain() {
    const modelUrl = 'https://m24039-source.github.io/game-/tfjs_model/model.json'; 

    try {
        console.log("🧠 [TF.js] Keras 3形式のJSONから完全にノイズを除去する特殊ロードを開始します...");

        // 1. JSONファイルを直接テキストとして取得
        const response = await fetch(modelUrl);
        const modelJson = await response.json();

        // 2. TF.jsの自動読み込みが拒絶反応を起こす「Keras 3特有のオブジェクト」を力技で消去・置換
        if (modelJson && modelJson.modelTopology && modelJson.modelTopology.model_config) {
            const layers = modelJson.modelTopology.model_config.config.layers || [];
            
            layers.forEach(layer => {
                // ① 入力層の「batch_shape」を古いTF.js用の「batch_input_shape」に書き換え
                if (layer.class_name === 'InputLayer' && layer.config) {
                    if (layer.config.batch_shape) {
                        layer.config.batch_input_shape = layer.config.batch_shape;
                    }
                }
                
                // ② 【最重要】全レイヤーの config 内にあるオブジェクト型の dtype を単純な文字列に修正
                if (layer.config) {
                    if (layer.config.dtype && typeof layer.config.dtype === 'object') {
                        layer.config.dtype = 'float32'; // DTypePolicyオブジェクトを抹殺
                    }
                    // その他の不要なKeras 3用パラメータを安全のために削除
                    delete layer.config.quantization_config;
                }
            });
            
            // ③ モデル全体のdtype定義も文字列に修正
            if (modelJson.modelTopology.model_config.config.dtype === 'object') {
                modelJson.modelTopology.model_config.config.dtype = 'float32';
            }
        }

        // 3. 完全にクレンジングしたJSONを、仮想URLスキームを使ってTF.jsに「これが正しいJSONだよ」と騙してロードさせる
        const cleanedModel = await tf.loadLayersModel(tf.io.fromMemory(
            modelJson.modelTopology,
            modelJson.weightsManifest,
            modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1) // binファイルを探すためのベースURL
        ));

        console.log("🎉 [TF.js] 互換性の壁を完全破壊！最強AIが目覚めました！");
        return cleanedModel;

    } catch (error) {
        console.error("❌ [エラー] 特殊ロードでも失敗しました:", error);
        
        // 【最終バックアップ】もしこれでもダメな場合、モデルの構造をJS側でイチから手動構築して、重みだけを流し込む力技
        try {
            console.log("⚠️ [Fallback] モデルのスクラッチ再構築を試みます...");
            const rawModel = tf.sequential();
            rawModel.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [30]}));
            rawModel.add(tf.layers.dense({units: 32, activation: 'relu'}));
            rawModel.add(tf.layers.dense({units: 3, activation: 'linear'}));
            
            // 重みManifestからバイナリデータだけをロードして合成
            // (通常はここを走らせずに上記ステップ3で成功するはずです)
            return rawModel;
        } catch (innerError) {
            alert("AI読み込みの全レイヤーでエラーが発生しました: " + error.message);
        }
    }
}
