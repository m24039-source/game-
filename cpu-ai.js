async function loadAIBrain() {
    // 脳みそファイルのURL
    const modelUrl = 'https://m24039-source.github.io/game-/tfjs_model/model.json'; 

    try {
        console.log("🧠 [TF.js] Keras3の構造を徹底的にクレンジングして読み込みます...");

        // 1. model.json を先読み
        const response = await fetch(modelUrl);
        const modelJson = await response.json();

        // 2. 【超強力クレンジング】古いTF.jsが拒絶反応を起こすKeras3固有のデータを消し去る
        if (modelJson && modelJson.modelTopology && modelJson.modelTopology.model_config) {
            const config = modelJson.modelTopology.model_config;
            const layers = config.config.layers || [];
            
            layers.forEach(layer => {
                // ① 入力層の形を古い形式に強制変換
                if (layer.class_name === 'InputLayer' && layer.config) {
                    if (layer.config.batch_shape && !layer.config.batch_input_shape) {
                        layer.config.batch_input_shape = layer.config.batch_shape;
                    }
                }
                // ② 【重要】全レイヤーから古いTF.jsが死ぬ原因になる「dtypeオブジェクト」を抹消
                if (layer.config) {
                    if (typeof layer.config.dtype === 'object') {
                        // オブジェクト型(DTypePolicy)になっていたら、ただの文字列 "float32" に上書き
                        layer.config.dtype = 'float32'; 
                    }
                    // その他不要なKeras3固有のパラメータを削除
                    delete layer.config.quantization_config;
                }
            });
        }

        // 3. 最もエラーが起きにくい「標準的なWebロード方式」に偽装してTF.jsに読み込ませる
        // メモリ展開(fromMemory)ではなく、パッチを当てたJSONを仮想的なURLとしてTF.jsに渡します
        const modifiedModel = await tf.loadLayersModel({
            load: async () => {
                return {
                    modelTopology: modelJson.modelTopology,
                    weightsManifest: modelJson.weightsManifest
                };
            },
            // 重みバイナリファイルの場所を教えてあげる
            path: modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1)
        });

        console.log("🎉 [TF.js] すべての呪いを解きました。AIのロードに完全成功しました！");
        return modifiedModel;

    } catch (error) {
        console.error("❌ [エラー] 最終クレンジングでもロードに失敗:", error);
        alert("AI読み込みエラー: " + error.message + "\n※シークレットタブで試してみてください。");
    }
}
