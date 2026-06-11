// ==========================================
// 数理変換タクティクス：CPU（AI）思考エンジン
// 【Google Colab・Python育成モデル 逆輸入版】
// ==========================================
import { runTransaction } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

// AIの脳みそ（ニューラルネットワークモデル）を保持する変数
let aiNeuralModel = null;

/**
 * 📥 1. Colabで育てた最強の脳みそ（model.json）を読み込む
 */
export async function loadBrain() {
    try {
        // 同じフォルダ階層にある tfjs_model/model.json をロード
        aiNeuralModel = await tf.loadLayersModel('./tfjs_model/model.json');
        console.log("🧠 [AI] Google Colabで鍛え上げた最強の脳みそを正常にロードしました！");
        return true;
    } catch (error) {
        console.error("⚠️ [AI] モデルのロードに失敗しました。ファイルが配置されているか確認してください:", error);
        return false;
    }
}

/**
 * 📊 2. 盤面データをAIが理解できる数値の配列（30要素）に正規化
 * (Python側の convert_state_to_vector と完全にロジックを一致させています)
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
    // もし脳みそがまだ読み込まれていなければ、自動でロードを試みる
    if (!aiNeuralModel) {
        loadBrain().then(success => {
            if (!success) {
                console.log("💤 AIの脳みそファイルが見つからないため、パス（手動初期化待ち）します。");
                return;
            }
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

        // --- 🧠 逆輸入した脳みそ（TensorFlow.js）による戦術推論 ---
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

        // 手番交代
        advanceTurn(currentData);
        resetTurnTimerStock(currentData);
        return currentData;
    });
}

// 共通ヘルパー関数
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
