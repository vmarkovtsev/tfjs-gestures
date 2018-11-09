const webcam = new Webcam(document.getElementById('webcam'));
const actionSamples = [];
const idleSamples = [];
isPredicting = false;


function getSample() {
    return tf.tidy(() => mobilenet.predict(webcam.capture()));
}


async function appendActionSample() {
    actionSamples.push(getSample());
    document.getElementById('action-counter').textContent = actionSamples.length;
}

async function appendIdleSample() {
    idleSamples.push(getSample());
    document.getElementById('idle-counter').textContent = idleSamples.length;
}

async function train() {
    await tf.nextFrame();
    await tf.nextFrame();
    isPredicting = false;
    // Bake the dataset
    const X = tf.concat2d(actionSamples.concat(idleSamples));
    const Y = tf.tidy(() => tf.stack(
        Array(actionSamples.length).fill(tf.tensor1d([1, 0])).concat(
        Array(idleSamples.length).fill(tf.tensor1d([0, 1])))
    ));

    // Creates a 2-layer fully connected model. By creating a separate model,
    // rather than adding layers to the mobilenet model, we "freeze" the weights
    // of the mobilenet model, and only train weights from the new model.
    model = tf.sequential({
      layers: [
        // Layer 1.
        tf.layers.dense({
            inputDim: featuresNumber,
            units: 100,
            activation: 'relu',
            kernelInitializer: 'varianceScaling',
            useBias: true
        }),
        // Layer 2. The number of units of the last layer should correspond
        // to the number of classes we want to predict.
        tf.layers.dense({
            units: 2,
            kernelInitializer: 'varianceScaling',
            useBias: false,
            activation: 'softmax'
        })
      ]
    });

    // Creates the optimizer which drives training of the model.
    const optimizer = tf.train.adam(0.0001);
    // We use categoricalCrossentropy which is the loss function we use for
    // categorical classification which measures the error between our predicted
    // probability distribution over classes (probability that an input is of each
    // class), versus the label (100% probability in the true class)>
    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

    // We parameterize batch size as a fraction of the entire dataset because the
    // number of examples that are collected depends on how many examples the user
    // collects. This allows us to have a flexible batch size.
    const batchSize = Math.floor(X.shape[0] * 0.4);
    if (!(batchSize > 0)) {
      throw new Error(`Too few training samples.`);
    }

    // Train the model! Model.fit() will shuffle X & Y so we don't have to.
    await model.fit(X, Y, {
      batchSize,
      epochs: 40,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
            document.getElementById('loss').textContent = 'Loss: ' + logs.loss.toFixed(5);
        }
      }
    });
    X.dispose();
    Y.dispose();
    isPredicting = true;
    predictLoop();
}

async function predictLoop() {
    while (isPredicting) {
        const predictedClassTensor = tf.tidy(() => {
            const predictions = model.predict(getSample());
            // Returns the index with the maximum probability. This number corresponds
            // to the class the model thinks is the most probable given the input.
            return predictions.as1D().argMax();
        });
        const predictedClass = (await predictedClassTensor.data())[0];
        predictedClassTensor.dispose();
        document.getElementById('status').textContent = ['ACTION', 'IDLE'][predictedClass];
        await tf.nextFrame();
    }
}

async function init() {
    try {
        await webcam.setup();
    } catch (e) {
        document.getElementById('no-webcam').style.display = 'block';
    }
    const rootPath = 'mobilenetv2_100_224/';
    mobilenet = await tf.loadFrozenModel(
        rootPath + 'tensorflowjs_model.pb',
        rootPath + 'weights_manifest.json');
    // Warm up the model. This uploads weights to the GPU and compiles the WebGL
    // programs so the first time we collect data from the webcam it will be
    // quick.
    const output = getSample();
    featuresNumber = output.shape[1];
    output.dispose();
}

// Initialize the application.
init();
