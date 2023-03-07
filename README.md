<p align="center">
  <img width="350" src="https://raw.githubusercontent.com/DrewJay/predator/master/logo.png">
</p>

# Predator v1.0.1 'Wild Ox 🐂'
Predator is a machine learning tool built on the top of TensorFlow.js, that significantly simplifies regression tasks. It is capable of doing simple regressions, multivariate regressions, linear or nonlinear, simplified model saving and retrieving and model performance visualizations (uses tfvis internally).

## Note
Predator has to be used on running server like webpack, since major browser's security policies do not allow to read files from serverless scripts. Predator needs this functionality to read CSV files.

## Why would I use this?
Predator saves time and implies cleaner code. You can train accurate neural network models using oneliners, or customize the training process using any of many configurations available. People with lack of knowledge of neural network mechanisms can create well working models as well, since the API is very intuitive and clear.

## Some examples
### House price prediction - The Hello World of machine learning universe
We use predator instances to train and use neural network models. This would be the simplest way to initialize predator instance:
```
const pred = new Predator({
    system: {
        visual: true,
        params: ['sqft_living', 'price'],
        csvPath: `../csv/dataset.csv`,
    },
});
```
We define system configuration, ```visual``` says that any training performed by <i>pred</i> instance should be graphically displayed, ```params``` specify what particular field we want to read from CSV file (in this case, we will train network to predict <i>price</i> field based on <i>sqft_living</i> field) and ```csvPath``` just specifies path to CSV file relative to predator.js file.

That's all, we are ready to train now!

Training is super easy:
```
const myModel = await pred.session('myModel');
```
That's it. Your model will be ready when training is finished. Model will also be saved as 'myModel' to local storage, as first parameter of ```session``` specifies. It can also be left out empty, no need to save anything.

Prediction is super easy as well:
```
const prediction = await pred.predict([1500]);
```
We tell Predator to use input value <i>1500</i> (living square feet, remember?) and predict the price.

Second parameter of <strong>predict</strong> can also contain key <i>name</i>, which specifies the model name of model stored in local storage. Therefore using ```{name: 'myModel'}```, ```{model: myModel}```, or  ```'myModel'``` would work the same. And that's why single Predator instance can use models trained by other instances and easily engage the prediction.

### 🦄 Use words (Experimental) 🦄
Predator also comes with another interesting way to describe your neural network - using fluent english text. Built-in Language Processing Unit (LPU) takes care of basic language parsing and is able to recognize
user demands up to particular extent. This would be completely legit way to define neural network in predator:

```
useWords("System configuration will be as follows -> set visual property to true, params property could be ['x','y'] and it would have csvpath property '../csv/dataset.csv'", Predator)
```

Second parameter of ```useWords``` is called ```consumer```. It is constructor meant to consume generated configuration. If omitted - the generated configuration itself is returned.

### A little more to know
This example used barely any constructor configurations. There are many options you can optimize the network with, but they can also be omitted and automatically defaulted. For example if you omit loss function, it defaults to <i>meanSquaredError</i>, or if you omit activation function, it defaults to <i>sigmoid</i>.

We use <strong>neural</strong> configuration to apply specific behavior on our neural networks:
```
const pred = new Predator({
    neural: {
        model: {
            epochs: 20,        // We will use 20 epochs for training.
            optimizer: 'sgd',  // Optimizer will be stochastic gradient descent.
        },
        layers: {
            bias: true,        // Each layer will use bias. 
            amount: 10,        // We will use 10 layers in total (including input and output layer).
            nodes: 6,          // 6 nodes per layer will be present.
        },
    },
    system: {
        visual: true,
        params: ['sqft_living', 'price'],
        csvPath: `../csv/dataset.csv`,
    },
});
```
All configurations available can be found below.

## Configurations

| Name          | Level     | Type                  | Default           | Example               | Info          |
|:------------- |:----------|:----------------------|:------------------|-----------------------|---------------|
| epochs        | model     | number                | 10                | epochs: 5             |
| loss          | model     | string                | meanSquaredError  | loss: 'logLoss'       |
| optimizer     | model     | string                | adam              | optimizer: 'sgd'      |
| ttSplit       | model     | number                | 2                 | ttSplit: 3            | test-train split
| bias          | layers    | boolean               | true              | bias: false           |
| activation    | layers    | string                | sigmoid           | activation: 'softmax' |
| amount        | layers    | number                | 3                 | amount: 5             | amount of layers (includes input and output layers)
| nodes         | layers    | number                | 10                | nodes: 16             | nodes per layer (excludes output layer)
| tensorShapes  | layers    | array[][]             | minimalShape*     | exampleShape*         | Custom tensor dimensions
| override      | layers    | array[]               |                   | override: [{}, ...]   | Custom array defining layers (see tensorflowJs API)
| visual        | system    | boolean               | false             | visual: true          | Display training statistics
| params        | system    | array[][] or array[]  |                   | params: ['a', 'b']    | Predict csv field 'b' based on field 'a'
| csvPath       | system    | string                |                   | csvPath: '../file.csv'|

<strong>*minimalShape = ```[[Predator.max(1), len(param[0])], [Predator.max(1), len(param[1])]]```</strong>

<strong>*exampleShape = ```[[Predator.max(2), 2], [Predator.max(1), 1]]```</strong>

When session is finished, ```generated``` level is created in configuration where specific session-generated and training information are stored. 

## Good practices
### Exception handling
Almost every function Predator instance offers is asynchronous and can return specific error, which makes debugging easier. It is therefore good
practice to surround sessions or predictions in try/catch block.

```
try {
    await pred.session('myModel');
} catch (PredatorException) {
    console.log(PredatorException.message);
    throw PredatorException;
}
```

Throwing the exception is very useful, since it can be analyzed from developer's console. Errors are designed to be very descriptive and suggest
solutions to specific problems, like tensor dimension adjustments.
