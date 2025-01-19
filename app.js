// モデルのパス
const MODEL_PATH = './model/model.json';

// ファイルアップロード要素
const imageInput = document.getElementById('imageInput');
const resultDiv = document.getElementById('result');

// モデルをロード
let model;
async function loadModel() {
  model = await tf.loadLayersModel(MODEL_PATH);
  console.log('モデルがロードされました');
}
loadModel();

// 画像がアップロードされたら処理する
imageInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  // 画像を読み込む
  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = async () => {
    // Tensorに変換
    const tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([224, 224]) // モデルに合わせてリサイズ
      .toFloat()
      .expandDims();

    // 推論を実行
    const predictions = await model.predict(tensor).data();

    // 結果を表示
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const labels = ['犬', '猫']; // metadata.jsonに合わせる
    resultDiv.innerHTML = `結果: ${labels[maxIndex]}（信頼度: ${(predictions[maxIndex] * 100).toFixed(2)}%）`;
  };
});
