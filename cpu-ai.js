// ==========================================
// 数理変換タクティクス：CPU（AI）思考エンジン
// 【ディープラーニング（TensorFlow.js）ニューラルネットモデル搭載版】
// ==========================================
import { runTransaction } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

// AIの脳みそ（ニューラルネットワークモデル）を保持する変数
let aiNeuralModel = null;

/**
 * 🧠 1. ディープラーニングの脳みそ（モデル）を初期化・作成する関数
 */
function createAiModel() {
    if (aiNeuralModel) return aiNeuralModel; // すでに作られていれば再利用

    // 3層のシンプルな全結合ニューラルネットワークを作成
    const model = tf.sequential();
    
    // 入力層：盤面の状態（自分の手札、相手の手札、スコアなど）を数値化したものを受け取る（今回は簡易的に30要素）
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [30] }));
    
    // 中間層：特徴を掛け合わせて複雑な状況を分析
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    
    // 出力層：各アクション（0:パス, 1:攻撃, 2:合成）の「期待度（価値）」を算出
    model.add(tf.layers.dense({ units: 3, activation: 'linear' }));

    // モデルのコンパイル（学習アルゴリズムの設定）
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    
    console.log("🤖 [DL-AI] TensorFlow.js ニューラルネットワークモデルが正常に生成されました。");
    return model;
}

/**
 * 📊 2. 現在のゲーム盤面（Firebaseのデータ）をAIの脳が理解できる「数値の配列（テンソル）」に変換する
 */
function convertStateToVector(currentData, cpuId) {
    const vector = new Array(30).fill(0); // 30マスの数字データ入れ
    
    if (!currentData || !currentData.players) return vector;

    // 自分の手札の情報を先頭10マスに入れる
    const myHand = targetHandArray(currentData.players[cpuId]?.hand);
    for (let i = 0; i < Math.min(myHand.length, 10); i++) {
        vector[i] = myHand[i] / 150.0; // 制限値150で割って 0〜1 の間に正規化
    }

    // 自分の現在のスコア
    vector[10] = (currentData.players[cpuId]?.score || 0) / 100.0;

    // 相手プレイヤーの手札情報（簡易的に最初の相手）
    const enemyId = Object.keys(currentData.players).find(pid => pid !== cpuId);
    if (enemyId) {
        const enemyHand = targetHandArray(currentData.players[enemyId]?.hand);
        for (let i = 0; i < Math.min(enemyHand.length, 10); i++) {
            vector[11 + i] = enemyHand[i] / 150.0;
        }
        vector[21] = (currentData.players[enemyId]?.score || 0) / 100.0;
    }

    // 現在のターン数など
    vector[22] = (currentData.turnCount || 1) / 10.0;

    return vector;
}

// ヘルパー関数: 配列への安全変換
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
        const maxTurnTarget = currentData.config?.maxTurns || 10;
        if (currentData.turnCount > maxTurnTarget) {
            currentData.status = 'finished';
            currentData.log = (currentData.log || "") + `🏁【制限ターン終了】ゲームを終了します。\n`;
            let highestScore = -1; let winnerName = "";
            Object.values(currentData.players).forEach(p => {
                if (p.score > highestScore) { highestScore = p.score; winnerName = p.name; }
            });
            currentData.log = (currentData.log || "") + `🏆 勝者: ${winnerName} (${highestScore} pt)\n`;
        }
    }
}

function resetTurnTimerStock(currentData) {
    currentData.remainingTime = currentData.config?.turnLimitTime || 30;
    currentData.timerMode = "stock"; 
}

/**
 * 🤖 3. DL対応型 CPU自動思考ロジック本体
 */
export function executeCPUTurn(roomRef, cpuId) {
    // 脳みそモデルの準備
    aiNeuralModel = createAiModel();

    runTransaction(roomRef, (currentData) => {
        if (!currentData || currentData.status !== 'playing') return currentData;
        
        const activePlayerId = currentData.turnOrder ? currentData.turnOrder[currentData.currentTurnIdx] : null;
        if (activePlayerId !== cpuId) return currentData;

        // ターン開始ドロー
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

        // --- 🧠 ディープラーニング（推論）の実行部分 ---
        // 盤面を数値ベクトル化
        const stateVector = convertStateToVector(currentData, cpuId);
        
        // TensorFlowのテンソル形式に変換し、脳みそに入力して予測値を出す
        let predictedActionCategory = 0; // 0: パス・合成、1: 攻撃
        
        tf.tidy(() => {
            const inputTensor = tf.tensor2d([stateVector]);
            const prediction = aiNeuralModel.predict(inputTensor);
            const actionScores = prediction.dataSync(); // [パス期待度, 攻撃期待度, 合成期待度] みたいな配列が返る
            
            // 最もスコア（期待度）が高い行動のカテゴリ（0か1か2）を選ぶ
            predictedActionCategory = prediction.argMax(-1).dataSync()[0];
            console.log(`🤖 [DL-AI思考中] 脳内の予測スコア: パス=${actionScores[0].toFixed(2)}, 攻撃=${actionScores[1].toFixed(2)}, 合成=${actionScores[2].toFixed(2)} -> 選択カテゴリ: ${predictedActionCategory}`);
        });

        // ルールベースの探索ロジックとディープラーニングの予測を融合（ハイブリッド思考）
        let bestAction = { type: "pass", score: -1, cardIdx: -1, card1Idx: -1, card2Idx: -1, resVal: -1, resVal2: -1, opLabel: "" };

        // ディープラーニングが「今は攻撃すべき（カテゴリ1）」と判断、または1周目でなければ攻撃を模索
        if (predictedActionCategory === 1 && !isFirstRound) {
            for (let i = 0; i < cpuHand.length; i++) {
                let attackNum = cpuHand[i];
                if (attackNum === 1) continue;
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

        // 攻撃しなかった、または脳が「合成・パスすべき」と判断した場合
        if (bestAction.type === "pass") {
            // 2枚合成の全探索
            if (cpuHand.length >= 2) {
                for (let i = 0; i < cpuHand.length; i++) {
                    for (let j = 0; j < cpuHand.length; j++) {
                        if (i === j) continue;
                        let n1 = cpuHand[i]; let n2 = cpuHand[j];
                        let addRes = n1 + n2;
                        if (addRes <= limit && addRes > bestAction.resVal) {
                            bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: addRes, opLabel: "➕ 足し算" };
                        }
                        let subRes = n1 - n2;
                        if (subRes > 0 && subRes > bestAction.resVal) {
                            bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: subRes, opLabel: "➖ 引き算" };
                        }
                    }
                }
            }
        }

        // --- 4. 行動結果のFirebase反映 ---
        if (bestAction.type === "attack") {
            let attackNum = bestAction.value;
            Object.keys(currentData.players).forEach(pid => {
                if (pid === cpuId) return;
                let targetHand = targetHandArray(currentData.players[pid].hand);
                let kept = [];
                targetHand.forEach(cardNum => {
                    if (cardNum % attackNum === 0) {
                        currentData.log = (currentData.log || "") + ` 🎯 ${currentData.players[pid].name} の [${cardNum}] を撃破！\n`;
                    } else { kept.push(cardNum); }
                });
                currentData.players[pid].hand = kept;
            });
            cpuHand.splice(bestAction.cardIdx, 1);
            currentData.players[cpuId].hand = cpuHand;
            currentData.players[cpuId].score += bestAction.score;
            currentData.log = (currentData.log || "") + `🧠 [DL思考] AIは攻撃を選択: [${attackNum}] で ${bestAction.score} 点強奪！\n`;
            
            const winTarget = currentData.config?.winScore || 100;
            if (currentData.players[cpuId].score >= winTarget) {
                currentData.status = 'finished';
                currentData.log = (currentData.log || "") + `🏆👑 AIが目標点に達し勝利しました！\n`;
            }
        } else if (bestAction.type === "op2") {
            let v1 = cpuHand[bestAction.card1Idx]; let v2 = cpuHand[bestAction.card2Idx];
            let idx1 = cpuHand.indexOf(v1); if(idx1 !== -1) cpuHand.splice(idx1, 1);
            let idx2 = cpuHand.indexOf(v2); if(idx2 !== -1) cpuHand.splice(idx2, 1);
            cpuHand.push(bestAction.resVal);
            currentData.players[cpuId].hand = cpuHand;
            currentData.log = (currentData.log || "") + `🧠 [DL思考] AIは合成を選択: ${bestAction.opLabel} [${v1}]&[${v2}] ➡ [${bestAction.resVal}]\n`;
        } else {
            currentData.log = (currentData.log || "") + `💤 AIはパスしました。\n`;
        }

        advanceTurn(currentData);
        resetTurnTimerStock(currentData);
        return currentData;
    });
}
