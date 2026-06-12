// ==========================================
// 数理変換タクティクス：CPU（AI）思考エンジン
// 【GitHub Pages直接通信・完全勝訴版】
// ==========================================
import { runTransaction } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

// AIの脳みそ（ニューラルネットワークモデル）を保持する変数
let aiNeuralModel = null;
let isModelLoading = false;

/**
 * 📥 1. GitHub Pages上にある本物のmodel.jsonをURLから直接一本釣りする
 */
export async function loadBrain() {
    if (aiNeuralModel) return true; // すでにロード済みの場合はスキップ
    if (isModelLoading) return false;
    
    isModelLoading = true;
    try {
        // 💡 あなたのGitHub Pagesの絶対パスを直接叩いて、ファイルをロードさせます
        // 末尾にダミーのタイムスタンプ(?t=...)を付けることで、ブラウザの頑固なキャッシュを強制破壊します
        const modelUrl = "https://m24039-source.github.io/game-/tfjs_model/model.json?t=" + Date.now();
        
        console.log("📡 [AI] モデルの直接通信を開始します... URL:", modelUrl);
        
        // 正真正銘、本物のファイルを読み込み
        aiNeuralModel = await tf.loadLayersModel(modelUrl);

        console.log("🏆 [AI] 脳みそファイルの読み込みとネットワークの構築に完全成功しました！");
        isModelLoading = false;
        return true;
    } catch (error) {
        console.error("⚠️ [AI] ファイルの読み込みに失敗しました。パスやGitHubの公開状態を確認してください:", error);
        isModelLoading = false;
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
 * 🤖 3. CPU自動思考ロジック本体
 */
export function executeCPUTurn(roomRef, cpuId) {
    if (!aiNeuralModel) {
        loadBrain(); 
        console.log("💤 AIの脳みそが準備中のため、この手番は安全にスキップします。");
        return;
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
        
        tf.tidy(() => {
            const inputTensor = tf.tensor2d([stateVector]);
            const prediction = aiNeuralModel.predict(inputTensor);
            predictedActionCategory = prediction.argMax(-1).dataSync()[0];
        });

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

        // カテゴリ2：高度な合成
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
