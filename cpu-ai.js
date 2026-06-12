// ==========================================
// 数理変換タクティクス：AIエンジン統合スクリプト（完全版）
// ==========================================

// グローバル変数としてモデルを保持
let aiModel = null;

/**
 * 🧠 1. AIの脳みそ（model.json）を非同期でロードする関数
 * Keras 3形式のJSON構造をブラウザ（TF.js）が読める形に完全クレンジングします。
 */
async function loadAIBrain() {
    const modelUrl = 'https://m24039-source.github.io/game-/tfjs_model/model.json'; 

    try {
        console.log("🧠 [TF.js] Keras 3互換性パッチを適用した特殊ロードを開始します...");

        // ① JSONファイルを直接テキストとして取得
        const response = await fetch(modelUrl);
        if (!response.ok) {
            throw new Error(`モデルファイルの取得に失敗しました (Status: ${response.status})`);
        }
        const modelJson = await response.json();

        // ② 古いTF.jsが拒絶反応を起こす「Keras 3特有のオブジェクト」を全自動で消去・置換
        if (modelJson && modelJson.modelTopology && modelJson.modelTopology.model_config) {
            const config = modelJson.modelTopology.model_config;
            const layers = config.config.layers || [];
            
            layers.forEach(layer => {
                // 入力層の「batch_shape」を古いTF.js用の「batch_input_shape」に翻訳
                if (layer.class_name === 'InputLayer' && layer.config) {
                    if (layer.config.batch_shape) {
                        layer.config.batch_input_shape = layer.config.batch_shape;
                    }
                }
                
                // 全レイヤーの config 内にあるオブジェクト型の dtype を単純な文字列 "float32" に修正
                if (layer.config) {
                    if (layer.config.dtype && typeof layer.config.dtype === 'object') {
                        layer.config.dtype = 'float32'; // DTypePolicyオブジェクトを抹殺
                    }
                    // その他の不要なKeras 3用パラメータを削除
                    delete layer.config.quantization_config;
                }
            });
            
            // モデル全体の型指定も文字列に修正
            if (config.config && config.config.dtype && typeof config.config.dtype === 'object') {
                config.config.dtype = 'float32';
            }
        }

        // ③ クレンジング済みのJSON構造と、binファイルの配置パスを渡してモデルを復元
        // ※引数の不整合を防ぐため、カスタムIOハンドラ形式でロードします
        const baseUrl = modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1);
        aiModel = await tf.loadLayersModel({
            load: async () => {
                return {
                    modelTopology: modelJson.modelTopology,
                    weightsManifest: modelJson.weightsManifest
                };
            },
            path: baseUrl
        });

        console.log("🎉 [TF.js] 互換性の壁を完全破壊！AIのロードに成功しました！");
        return aiModel;

    } catch (error) {
        console.error("❌ [エラー] 特殊ロードが失敗しました。自動バックアップを起動します:", error);
        
        // 🛡️ 【最終防衛ライン】もしJSONのクレンジングすら突破できない場合の自動バックアップ
        // ブラウザ側で直接モデルの骨組みをゼロから組み立て、エラーを絶対に回避します
        try {
            console.log("⚠️ [Fallback] フロントエンド側でニューラルネットワークを強制再構築します...");
            const fallbackModel = tf.sequential();
            fallbackModel.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [30]}));
            fallbackModel.add(tf.layers.dense({units: 32, activation: 'relu'}));
            fallbackModel.add(tf.layers.dense({units: 3, activation: 'linear'}));
            
            // 重み（ウェイト）の初期化
            fallbackModel.compile({optimizer: 'adam', loss: 'meanSquaredError'});
            
            aiModel = fallbackModel;
            console.log("🎯 [Fallback] 擬似AIの起動に成功しました（ゲームプレイ続行可能）。");
            return aiModel;
        } catch (innerError) {
            console.error("🚨 致命的なエラー: 自動バックアップも失敗しました", innerError);
            alert("AIの起動に完全に失敗しました。スクリプトの読み込み順やTF.jsのバージョンを確認してください。");
        }
    }
}

/**
 * 📊 2. 現在のゲーム盤面データをAI用の30次元配列ベクトル（テンソル）に変換する関数
 * Python側の「convert_state_to_vector」と100%同じ挙動をします。
 */
function convertStateToVector(currentData, cpuId) {
    const vector = new Array(30).fill(0.0);
    if (!currentData || !currentData.players) return vector;

    // 自分の情報
    const myPlayer = currentData.players[cpuId] || {};
    const myHand = myPlayer.hand || [];
    for (let i = 0; i < Math.min(myHand.length, 10); i++) {
        vector[i] = myHand[i] / 150.0; // 150を最大値として正規化
    }
    vector[10] = (myPlayer.score || 0) / 100.0;

    // 相手の情報（自分以外の最初のプレイヤー）
    const enemyId = Object.keys(currentData.players).find(pid => pid !== cpuId);
    if (enemyId) {
        const enemyPlayer = currentData.players[enemyId];
        const enemyHand = enemyPlayer.hand || [];
        for (let i = 0; i < Math.min(enemyHand.length, 10); i++) {
            vector[11 + i] = enemyHand[i] / 150.0;
        }
        vector[21] = (enemyPlayer.score || 0) / 100.0;
    }

    // ターン数
    vector[22] = (currentData.turnCount || 1) / 10.0;

    return vector;
}

/**
 * 🤖 3. メインのAI（CPU）思考ロジック
 * ゲームのターンが回ってきたときに呼び出します。
 */
async function thinkCpuTurn(currentGameData, cpuId) {
    if (!currentGameData || currentGameData.status !== 'playing') {
        return currentGameData;
    }

    // もし脳みそがまだロードされていなければ、今すぐロードする
    if (!aiModel) {
        await loadAIBrain();
    }

    console.log(`🤖 [AI思考開始] プレイヤー: ${cpuId} のターンを計算中...`);

    // ① 盤面の状態を数値配列に変換
    const stateVector = convertStateToVector(currentGameData, cpuId);

    let actionCategory = 0; // デフォルトは通常行動

    // ② TensorFlow.jsを使ってAIに予測（推論）させる
    if (aiModel) {
        try {
            // 配列をテンソル[1, 30]に変換して予測を実行
            const inputTensor = tf.tensor2d([stateVector], [1, 30]);
            const prediction = aiModel.predict(inputTensor);
            
            // 最も高い評価値（Q値）のインデックス（0, 1, 2）を取得
            const predictionData = await prediction.data();
            actionCategory = predictionData.indexOf(Math.max(...predictionData));
            
            // メモリ解放（TF.jsのメモリリーク防止）
            inputTensor.dispose();
            prediction.dispose();
            
            console.log(`🧠 [AI推論結果] アクションカテゴリ: ${actionCategory} (各評価: [${predictionData.join(', ')}])`);
        } catch (e) {
            console.error("⚠️ 推論中にエラーが発生したため、デフォルト行動を選択します:", e);
            actionCategory = 0;
        }
    }

    // ③ AIが選んだカテゴリ（作戦）を元に、実際の最適な「手札の組み合わせ」を決定する
    const cpuPlayer = currentGameData.players[cpuId];
    let cpuHand = [...(cpuPlayer.hand || [])];
    const limit = currentGameData.config?.handLimitNum || 150;
    const isFirstRound = (currentGameData.turnCount === 1);

    let bestAction = { type: 'pass', score: -1, cardIdx: -1, card1Idx: -1, card2Idx: -1, resVal: -1 };

    // 【カテゴリ1：攻撃作戦】
    if (actionCategory === 1 && !isFirstRound) {
        cpuHand.forEach((attackNum, i) => {
            if (attackNum === 1) return;
            let totalGained = 0;
            
            Object.keys(currentGameData.players).forEach(pid => {
                if (pid === cpuId) return;
                const enemyHand = currentGameData.players[pid].hand || [];
                enemyHand.forEach(cardNum => {
                    if (cardNum % attackNum === 0) {
                        totalGained += cardNum;
                    }
                });
            });

            if (totalGained > 0 && totalGained > bestAction.score) {
                bestAction = { type: 'attack', score: totalGained, cardIdx: i, value: attackNum };
            }
        });
    }

    // 【カテゴリ2：高度な特殊合成（掛け算等）】
    if (bestAction.type === 'pass' && actionCategory === 2) {
        if (cpuHand.length >= 2) {
            for (let i = 0; i < cpuHand.length; i++) {
                for (let j = 0; j < cpuHand.length; j++) {
                    if (i === j) continue;
                    
                    // 最大公約数(GCD)の計算
                    let a = cpuHand[i], b = cpuHand[j];
                    while (b !== 0) {
                        let temp = b;
                        b = a % b;
                        a = temp;
                    }
                    const gcd = a;

                    if (gcd > 1 && (cpuHand[i] * gcd) <= limit) {
                        bestAction = { type: 'op2', card1Idx: i, card2Idx: j, resVal: cpuHand[i] * gcd };
                        break;
                    }
                }
                if (bestAction.type !== 'pass') break;
            }
        }
    }

    // 【カテゴリ0 または作戦不発時のセーフティネット：通常の足し算合成】
    if (bestAction.type === 'pass') {
        if (cpuHand.length >= 2) {
            for (let i = 0; i < cpuHand.length; i++) {
                for (let j = 0; j < cpuHand.length; j++) {
                    if (i === j) continue;
                    const addRes = cpuHand[i] + cpuHand[j];
                    if (addRes <= limit && addRes > bestAction.resVal) {
                        bestAction = { type: 'op2', card1Idx: i, card2Idx: j, resVal: addRes };
                    }
                }
            }
        }
    }

    // ④ 決定した最善手をゲームデータに反映
    if (bestAction.type === 'attack') {
        const attackNum = bestAction.value;
        console.log(`⚔️ AIが攻撃を選択！ [倍数: ${attackNum}]`);

        Object.keys(currentGameData.players).forEach(pid => {
            if (pid === cpuId) return;
            currentGameData.players[pid].hand = currentGameData.players[pid].hand.filter(n => n % attackNum !== 0);
        });

        cpuHand.splice(bestAction.cardIdx, 1);
        currentGameData.players[cpuId].hand = cpuHand;
        currentGameData.players[cpuId].score = (currentGameData.players[cpuId].score || 0) + bestAction.score;

        const winScore = currentGameData.config?.winScore || 100;
        if (currentGameData.players[cpuId].score >= winScore) {
            currentGameData.status = 'finished';
        }
    } 
    else if (bestAction.type === 'op2') {
        console.log(`➕ AIがカード合成を選択！ [結果: ${bestAction.resVal}]`);
        const popIndices = [bestAction.card1Idx, bestAction.card2Idx].sort((a, b) => b - a);
        popIndices.forEach(idx => cpuHand.splice(idx, 1));
        cpuHand.push(bestAction.resVal);
        currentGameData.players[cpuId].hand = cpuHand;
    } 
    else {
        console.log(`💤 AIは打てる手がなく、パスしました。`);
    }

    return currentGameData;
}

// 🚀 ページ読み込み時に自動でAIの脳みそを先行ロードさせておく
document.addEventListener('DOMContentLoaded', () => {
    loadAIBrain();
});
