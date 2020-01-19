# Predator
Predator is a tool built on the top of TensorFlow.js, that significantly simplifies predictions (also called regression tasks). It is capable of doing simple regressions, multivariate regressions, linear or nonlinear, simplified model saving and retrieving and model performance visualizations (uses tfvis internally).

## Why would I use this?
Predator saves time, and implies cleaner code. You can train accurate neural network models using oneliners, or custommize the training process using any of many configurations available. People with lack of knowledge of neural network mechanisms can create well working models as well, since the API is very intuitive and clear.

## Some examples
We use predator instances to train and use neural network models. This would be the simplest way to initialize predator instance:
```
const pred = new Predator({
                system: {
                    visual: true,
                    params: ['sqft_living', 'price'],
                    csvPath: `../csv/dataset.csv`,
                }
            });
```
We define system configuration, <strong>visual</strong> says that any training performed by <i>pred</i> instance should be graphically displayed, <strong>params</strong> specify what particular field we want to read from CSV file (in this case, we will train network to predict <i>price</i> field based on <i>sqft_living</i> field) and <strong>csvPath</strong> just specifies path to CSV file relative to predator.js file.

That's all, we are ready to train now!

Training is super easy:
```
const myModel = await pred.session('myModel');
```
That's it. Your model will be ready when training is finished. Model also will be saved as 'myModel' to local storage, as first parameter of <strong>session</strong> specifies. It can also be left out empty, no need to save anything.

Prediction is super easy as well:
```
const prediction = await pred.predict([1500], {model});
```
We tell Predator to use input value <i>1500</i> (living square feet, remember?) and predict the price.

Second parameter of <strong>predict</strong> can also contain key <i>name</i>, which specifies the model name of model stored in local storage. Therefore using ```{name: 'myModel'}``` would work the same. And that's why single Predator instance can use models trained by other instances and easily engage the prediction.

### A little more to know
This example used barely any constructor configurations. There are many options you can optimize the network with, but they can also be omitted and automatically defaulted. For example if you ommit error function, it defaults to <i>meanSquaredError</i>, or if you omit activation function, it defaults to <i>sigmoid</i>.