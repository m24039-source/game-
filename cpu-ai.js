// ==========================================
// 数理変換タクティクス：CPU（AI）思考エンジン
// 【最強脳みそデータ直接埋め込み・最終安定版】
// ==========================================
import { runTransaction } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

// AIの脳みそ（ニューラルネットワークモデル）を保持する変数
let aiNeuralModel = null;

/**
 * 📥 1. 脳みその構造データを直接読み込む
 */
export async function loadBrain() {
    try {
        // 先ほど提示していただいた正しいmodel.jsonのデータを直接変数に格納します
        const modelJsonData = {
            "format": "layers-model", 
            "generatedBy": "keras v3.13.2", 
            "convertedBy": "TensorFlow.js Converter v4.22.0", 
            "modelTopology": {
                "keras_version": "3.13.2", 
                "backend": "tensorflow", 
                "model_config": {
                    "class_name": "Sequential", 
                    "config": {
                        "name": "sequential", 
                        "trainable": true, 
                        "layers": [
                            {"class_name": "InputLayer", "config": {"batch_shape": [null, 30], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer", "optional": false}}, 
                            {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "units": 64, "activation": "relu", "use_bias": true}}, 
                            {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "units": 32, "activation": "relu", "use_bias": true}}, 
                            {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "units": 3, "activation": "linear", "use_bias": true}}
                        ], 
                        "build_input_shape": [null, 30]
                    }
                }
            }, 
            "weightsManifest": [{
                "paths": ["https://m24039-source.github.io/game-/tfjs_model/group1-shard1of1.bin"], 
                "weights": [
                    {"name": "sequential/dense/kernel", "shape": [30, 64], "dtype": "float32"}, 
                    {"name": "sequential/dense/bias", "shape": [64], "dtype": "float32"}, 
                    {"name": "sequential/dense_1/kernel", "shape": [64, 32], "dtype": "float32"}, 
                    {"name": "sequential/dense_1/bias", "shape": [32], "dtype": "float32"}, 
                    {"name": "sequential/dense_2/kernel", "shape": [32, 3], "dtype": "float32"}, 
                    {"name": "sequential/dense_2/bias", "shape": [3], "dtype": "float32"}
                ]
            }]
        };
        
        // ファイル経由ではなく、上記のデータから直接TensorFlow.jsのモデルを復元
        aiNeuralModel = await tf.loadLayersModel(tf.io.fromMemory(
            modelJsonData.modelTopology,
            modelJsonData.weightsManifest
        ));

        console.log("🧠 [AI] 埋め込み脳みそのロードに成功しました！");
        return true;
    } catch (error) {
        console.error("⚠️ [AI] 脳みその復元に失敗しました:", error);
        alert("🚨 エラー詳細:\n" + error.message);
        return false;
    }
}

/**
 * 📊 2. 盤面データをAIが理解できる数値の配列（30要素）に正規化
 */
function convertStateToVector(currentData, cpuId) {
    const vector = new Array(30).fill(0);
    if (!currentData || !currentData.players) return vector;

    const myHand = targetHandArray(currentData.players[cpuId]?.hand);
    for (let i = 0; i < Math.min(myHand.length, 10); i++) {
        vector[i] = myHand[i] / 150.0; 
    }
    vector[10] = (currentData.players[cpuId]?.score || 0) / 100.0;

    const enemyId = Object.keys(currentData.players).find(pid => pid !== cpuId);
    if (enemyId) {
        const enemyHand = targetHandArray(currentData.players[enemyId]?.hand);
        for (let i = 0; i < Math.min(enemyHand.length, 10); i++) {
            vector[11 + i] = enemyHand[i] / 150.0;
        }
        vector[21] = (currentData.players[enemyId]?.score || 0) / 100.0;
    }
    vector[22] = (currentData.turnCount || 1) / 10.0;
    return vector;
}

/**
 * 🤖 3. コラボーレーション型 CPU自動思考ロジック本体
 */
export function executeCPUTurn(roomRef, cpuId) {
    if (!aiNeuralModel) {
        loadBrain().then(success => {
            if (!success) return;
        });
    }

    runTransaction(roomRef, (currentData) => {
        if (!currentData || currentData.status !== 'playing') return currentData;
        
        const activePlayerId = currentData.turnOrder ? currentData.turnOrder[currentData.currentTurnIdx] : null;
        if (activePlayerId !== cpuId) return currentData;

        // ドロー処理
        if (currentData.absoluteTurnIdx > 0) {
            const rule = currentData.config || { drawCount: 2, drawMaxNum: 20 };
            let cpuHand = targetHandArray(currentData.players[cpuId].hand);
            let drawnNums = [];
            for(let i = 0; i < rule.drawCount; i++) {
                let drawn = Math.floor(Math.random() * rule.drawMaxNum) + 1;
                cpuHand.push(drawn);
                drawnNums.push(`[${drawn}]`);
            }
            currentData.players[cpuId].hand = cpuHand;
            currentData.log = (currentData.log || "") + `📥 AIドロー: ${currentData.players[cpuId].name} が ${drawnNums.join(', ')} を引きました。\n`;
        }

        let cpuHand = targetHandArray(currentData.players[cpuId].hand);
        let limit = currentData.config?.handLimitNum || 150;
        let isFirstRound = (currentData.turnCount === 1);

        // --- 🧠 逆輸入した脳みそによる戦術推論 ---
        const stateVector = convertStateToVector(currentData, cpuId);
        let predictedActionCategory = 0; 
        
        if (aiNeuralModel) {
            tf.tidy(() => {
                const inputTensor = tf.tensor2d([stateVector]);
                const prediction = aiNeuralModel.predict(inputTensor);
                predictedActionCategory = prediction.argMax(-1).dataSync()[0];
            });
        }

        // 最善手決定用変数
        let bestAction = { type: "pass", score: -1, cardIdx: -1, card1Idx: -1, card2Idx: -1, resVal: -1, opLabel: "" };

        // カテゴリ1：攻撃
        if (predictedActionCategory === 1 && !isFirstRound) {
            for (let i = 0; i < cpuHand.length; i++) {
                let attackNum = cpuHand[i]; if (attackNum === 1) continue;
                let totalGained = 0;
                Object.keys(currentData.players).forEach(pid => {
                    if (pid === cpuId) return;
                    let targetHand = targetHandArray(currentData.players[pid].hand);
                    targetHand.forEach(cardNum => { if (cardNum % attackNum === 0) totalGained += cardNum; });
                });
                if (totalGained > 0 && totalGained > bestAction.score) {
                    bestAction = { type: "attack", score: totalGained, cardIdx: i, value: attackNum };
                }
            }
        }

        // カテゴリ2：高度な合成（操作C）
        if (bestAction.type === "pass" && predictedActionCategory === 2) {
            if (cpuHand.length >= 2) {
                for (let i = 0; i < cpuHand.length; i++) {
                    for (let j = 0; j < cpuHand.length; j++) {
                        if (i === j) continue;
                        let a = cpuHand[i], b = cpuHand[j];
                        while (b !== 0) { let t = b; b = a % b; a = t; }
                        if (a > 1 && (cpuHand[i] * a) <= limit) {
                            bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: cpuHand[i] * a, opLabel: "🔮 操作C: 最大公約数掛け" };
                            break;
                        }
                    }
                    if (bestAction.type !== "pass") break;
                }
            }
        }

        // カテゴリ0 またはセーフティネット（通常足し算合成）
        if (bestAction.type === "pass") {
            if (cpuHand.length >= 2) {
                for (let i = 0; i < cpuHand.length; i++) {
                    for (let j = 0; j < cpuHand.length; j++) {
                        if (i === j) continue;
                        let addRes = cpuHand[i] + cpuHand[j];
                        if (addRes <= limit && addRes > bestAction.resVal) {
                            bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: addRes, opLabel: "➕ 足し算" };
                        }
                    }
                }
            }
        }

        // --- 実行の反映 ---
        if (bestAction.type === "attack") {
            let attackNum = bestAction.value;
            Object.keys(currentData.players).forEach(pid => {
                if (pid === cpuId) return;
                let targetHand = targetHandArray(currentData.players[pid].hand);
                let kept = [];
                targetHand.forEach(cardNum => {
                    if (cardNum % attackNum === 0) {
                        currentData.log = (currentData.log || "") + ` 🎯 AIが ${currentData.players[pid].name} の [${cardNum}] を撃破！\n`;
                    } else { kept.push(cardNum); }
                });
                currentData.players[pid].hand = kept;
            });
            cpuHand.splice(bestAction.cardIdx, 1);
            currentData.players[cpuId].hand = cpuHand;
            currentData.players[cpuId].score += bestAction.score;
            currentData.log = (currentData.log || "") + `🧠 AI[Colab脳選択]: [${attackNum}] で攻撃、${bestAction.score}点強奪！\n`;
            
            const winTarget = currentData.config?.winScore || 100;
            if (currentData.players[cpuId].score >= winTarget) {
                currentData.status = 'finished';
                currentData.log = (currentData.log || "") + `🏆 最終勝者は AI です！\n`;
            }
        } else if (bestAction.type === "op2") {
            let v1 = cpuHand[bestAction.card1Idx]; let v2 = cpuHand[bestAction.card2Idx];
            let idx1 = cpuHand.indexOf(v1); if(idx1 !== -1) cpuHand.splice(idx1, 1);
            let idx2 = cpuHand.indexOf(v2); if(idx2 !== -1) cpuHand.splice(idx2, 1);
            cpuHand.push(bestAction.resVal);
            currentData.players[cpuId].hand = cpuHand;
            currentData.log = (currentData.log || "") + `🧠 AI[Colab脳選択]: ${bestAction.opLabel} ([${v1}]&[${v2}] ➡ [${bestAction.resVal}])\n`;
        } else {
            currentData.log = (currentData.log || "") + `💤 AIはパスしました。\n`;
        }

        advanceTurn(currentData);
        resetTurnTimerStock(currentData);
        return currentData;
    });
}

function targetHandArray(handData) {
    if (!handData) return [];
    if (Array.isArray(handData)) return handData;
    return Object.values(handData);
}

function advanceTurn(currentData) {
    currentData.currentTurnIdx = (currentData.currentTurnIdx + 1) % currentData.turnOrder.length;
    currentData.absoluteTurnIdx++; 
    if (currentData.currentTurnIdx === 0) {
        currentData.turnCount++;
        if (currentData.turnCount > (currentData.config?.maxTurns || 10)) {
            currentData.status = 'finished';
        }
    }
}

function resetTurnTimerStock(currentData) {
    currentData.remainingTime = currentData.config?.turnLimitTime || 30;
    currentData.timerMode = "stock"; 
}
