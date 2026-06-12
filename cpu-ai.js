async function loadAIBrain() {
    // 脳みそファイルのURL
    const modelUrl = 'https://m24039-source.github.io/game-/tfjs_model/model.json'; 

    try {
        console.log("🧠 [TF.js] 実際のJSON構造に合わせた互換性パッチを適用して読み込みます...");

        // 1. model.json を一度テキストとして先読み
        const response = await fetch(modelUrl);
        const modelJson = await response.json();

        // 2. Keras 3特有の「batch_shape」を、古いTF.jsが理解できる「batch_input_shape」に翻訳するパッチ
        if (modelJson && modelJson.modelTopology && modelJson.modelTopology.model_config) {
            const layers = modelJson.modelTopology.model_config.config.layers || [];
            
            layers.forEach(layer => {
                if (layer.class_name === 'InputLayer' && layer.config) {
                    // もし batch_shape があって batch_input_shape が無ければ、値をコピーしてあげる
                    if (layer.config.batch_shape && !layer.config.batch_input_shape) {
                        layer.config.batch_input_shape = layer.config.batch_shape;
                        console.log("🔧 [Patch] InputLayerの batch_shape を batch_input_shape に変換しました:", layer.config.batch_input_shape);
                    }
                }
            });
        }

        // 3. パッチをあてたデータを元に、TF.jsに直接モデルを構築させる
        const model = await tf.loadLayersModel(tf.io.fromMemory(
            modelJson.modelTopology,
            modelJson.weightsManifest,
            modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1) // 重みファイル(.bin)を探すベースURL
        ));

        console.log("🎉 [TF.js] 互換性の壁を突破！AIの脳みそが完全にロードされました！");
        return model;

    } catch (error) {
        console.error("❌ [エラー] モデルのロードに失敗しました:", error);
        alert("モデルの読み込みエラー: " + error.message);
    }
}
