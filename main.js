// defining my model
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Compilation of my model
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});


// Generate some synthetic data for training
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train the model
async function train() {
  for (let i = 0; i < 100; i++) {
    const response = await model.fit(xs, ys, {epochs: 10});
    console.log(response.history.loss[0]);
  }
}

// Prediction code here
function predict() {
    const xInput = document.getElementById('x').value;
    
    if (xInput.trim() === '' || isNaN(parseFloat(xInput))) {
      alert('Please enter a valid digit.');
    } else {
      const x = tf.tensor2d([[parseFloat(xInput)]], [1, 1]);
      const result = model.predict(x);
      document.getElementById('result').innerText = `Predicted y value: ${result.dataSync()[0]}`;
      document.getElementById('result').style.padding = '1rem';
      document.getElementById('result').style.backgroundColor = '#1e1e21';
      document.getElementById('result').style.color = '#fff'
    }
}

window.onload = train;
