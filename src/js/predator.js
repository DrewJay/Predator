/**
 * Pred(ictor)ator class is construct simplifying
 * all kinds of regressions and neural network model
 * creation. Based on tensorflow js.
 */
const predator = function(config) {

    // - neural/model:
    //      + epochs {number}
    //      + loss {string}
    //      + optimizer {string}
    //      + shapeScale {number}
    //      + split {number}
    // - neural/layers
    //      + bias {boolean}
    //      + activation {string}
    //      + amount {number}
    //      + nodes {number}
    //      + tensorShape {array}
    // - system
    //      + params {array}
    //      + csvPath {string}
    this.config = config;

    this.config.generated = {};
    this.predCache = [];
    this.points = [];

    /**
     * Combine multiple generic plots. YES
     * 
     * @param modelData - Object containing model name or model itself
     * @param shouldAggregate - If should simulate synthetic state
     * @param shouldPredict - If should render prediction line
     */
    this.mergePlot = async (modelData, shouldAggregate, shouldPredict) => {

        if (!tfvis.visor().isOpen()) { tfvis.visor().toggle(); }

        const model = await predator.unpackModel(modelData, modelData.noModelCallback);
        if (!model && shouldPredict) { return model; }

        if (shouldAggregate) {
            await this.aggregateState(model.modelName, this);
        }

        const predictedPoints = (shouldPredict) ? await this.generatePredictionPoints({ model }, this) : [];
        const pointArrays = [this.points];
        const seriesArrays = ['original'];

        if(predictedPoints) {
            pointArrays.push(predictedPoints);
            seriesArrays.push('predicted');
        }

        predator.genericPlot([this.points, predictedPoints], ['original', 'predicted'], modelData, this);
    }

    /**
     * Generate prediction points that form a line. YES
     * 
     * @param modelData - Object containing model name or model itself
     * @returns Array of points
     */
    this.generatePredictionPoints = async (modelData) => {
        const model = await predator.unpackModel(modelData, modelData.noModelCallback);
        let sampleSize = 100;
        const adjusted = this.config.generated.adjusted;
        
        if (!model) { return []; }

        const [xs, ys] = tf.tidy(() => {
            
            const maxer = predator.max(this.config.neural.layers.tensorShape.slice(1).reduce((a, b) => a * b));
            
            const tensorBias = (adjusted) ? 
                predator[adjusted.with](adjusted.using).fn(new Array(sampleSize * adjusted.using)) : 
                maxer.fn(new Array(sampleSize * maxer.param));

            sampleSize *= (adjusted) ? adjusted.using : maxer.param;

            const normalizedXs = tf.linspace(0, 1, sampleSize),
                normalizedYs = model.predict(normalizedXs.reshape([tensorBias, this.config.neural.layers.tensorShape.slice(1)].flat()));

            const dnxs = predator.denormalizeTensor(normalizedXs, this.predCache[0]);
                  dnys = predator.denormalizeTensor(normalizedYs, this.predCache[1]);

            return [ dnxs.dataSync(), dnys.dataSync() ];
        });

        return Array.from(xs).map((val, index) => {
            return { x: val, y: ys[index] };
        });
    }

    /**
     * Attempt to make a prediction. YES
     * 
     * @param modelData - Object containing model name or model itself
     * @param x - Feature value
     * @returns Predicted x value
     */
    this.predict = async (modelData, values) => {
        const model = await predator.unpackModel(modelData, modelData.noModelCallback);
        if (!model) { return null; }
        await this.aggregateState(model.modelName, this);
        const inputTensor = predator.normalizeTensor(predator.makeTensor(values, [this.config.neural.model.shapeScale || 1, this.config.neural.layers.tensorShape.slice(1)].flat()), this.predCache[0]);
        const outputTensor = predator.denormalizeTensor(model.predict(inputTensor), this.predCache[1]);
        return outputTensor.dataSync()[0];
    }

    /**
     * Aggregate particular training state so
     * we can use features it produces in process.
     * 
     * @param modelName - If used, state is aggregated from saved model.
     * @param instance - Predator instance
     */
    this.aggregateState = async (modelName) => {
        this.config = predator.getConfig(modelName);
        const params = this.config.system.params;
        this.points = await predator.consumeCSV(this.config.system.csvPath, params);
        this.predCache = [];
        await predator.tensorFromArray(this.config.neural.layers.tensorShape, this.points, 'x', this),
        await predator.tensorFromArray(this.config.neural.layers.tensorShape, this.points, 'y', this);
    }

    /**
     * Run training session. NO
     * 
     * @param name - Model name to save
     */
    this.session = async (name) => {

        // Reset predCache.
        this.predCache = [];

        // Read data from CSV.
        this.points = await predator.consumeCSV(this.config.system.csvPath, this.config.system.params);
        
        // Create feature and label tensors.
        const featureTensor = await predator.tensorFromArray(this.config.neural.layers.tensorShape, this.points, 'x', this),
              labelTensor = await predator.tensorFromArray(this.config.neural.layers.tensorShape, this.points, 'y', this);

        // Test-train split.
        const [trainFeatureTensor, testFeatureTensor] = tf.split(featureTensor, this.config.neural.split || 2),
              [trainLabelTensor, testLabelTensor] = tf.split(labelTensor, this.config.neural.split || 2);
        
        // Create tsfjs model and train it.
        const model = predator.createModel(
            predator.denseGenerator({ amount: this.config.neural.layers.amount, units: this.config.neural.layers.nodes, bias: this.config.neural.layers.bias, activation: this.config.neural.layers.activation }, this.config.neural.layers.tensorShape ), this.config.neural.model.optimizer, this.config.neural.model.loss
        );

        const trainResult = await predator.train(model, this.config.neural.model.epochs, { trainFeatureTensor, trainLabelTensor } );
        
        // If name is set, save the model.
        if (name) {
            await predator.saveModel(model, name, this.config);
        }

        // Calculate test loss.
        const lossTensor = model.evaluate(testFeatureTensor, testLabelTensor);
        const testLoss = await lossTensor.dataSync();
        
        // Plot the results.
        await this.mergePlot({ model: model, name }, false, true, pred);
        tfvis.render.barchart({ name: 'Test vs Train' }, [{ index: 'Train', value: trainResult.history.loss[this.config.neural.model.epochs - 1] }, { index: 'Test', value: testLoss }]);
    }
}

/**
 * Return length of an input.
 */
predator.max = function(divide = 1) {
    return {
        fn: (input) => input.length / divide,
        name: 'max',
        param: divide,
    }
}

/**
 * Create array containing ones from shape.
 * 
 * @param shape - Tensor shape
 * @returns Array containing ones
 */
predator.ones = (shape) => {
    return shape.map(() => 1);
};

/**
 * Normalize tensor values (downscaling). YES
 * 
 * @param tensor - Tensor object
 * @param override - Use min and max from this overriding tensor
 * @returns Normalized tensor
 */
predator.normalizeTensor = (tensor, override) => {
    const min = (override) ? override.min() : tensor.min();
    const max = (override) ? override.max() : tensor.max();
    return tensor.sub(min).div(max.sub(min));
}

/**
 * Denormalize tensor values (upscaling). YES
 * 
 * @param tensor - Tensor object
 * @param override - Use min and max from this overriding tensor
 * @returns Denormalized tensor
 */
predator.denormalizeTensor = (tensor, override) => {
    const min = (override) ? override.min() : tensor.min();
    const max = (override) ? override.max() : tensor.max();
    return tensor.mul(max.sub(min)).add(min);
}

/**
 * Create tsfjs model used for training and testing. YES
 * 
 * @param layers - Array of objects defining layers
 * @param optimizerName - String name of optimizer function
 * @param loss - Name of a loss function
 * @returns Compiled tsfjs model
 */
predator.createModel = (layers, optimizerName, loss) => {
    const model = tf.sequential();

    for (const layer of layers) {
        model.add(tf.layers.dense(layer));
    }

    const optimizer = tf.train[optimizerName]();

    model.compile({
        loss,
        optimizer
    });

    return model;
}

/**
 * Engage model training phase. YES
 * 
 * @param model - Tsfjs model reference
 * @param epochs - Amount of epochs
 * @param tensors - Feature and label tensors
 * @returns Training data
 */
predator.train = async (model, epochs, { trainFeatureTensor, trainLabelTensor }) => {

    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ['loss']
    );

    return await model.fit(trainFeatureTensor, trainLabelTensor, {
        epochs,
        callbacks: {
            onEpochEnd,
        },
    });
}

/**
 * Read CSV from gived URL. YES
 * 
 * @param url - Path to CSV
 * @param params - If present, override local params
 * @returns Array-ized CSV data
 */
predator.consumeCSV = async (url, params) => {
    const data = tf.data.csv(url);
    
    const pointDataSet = await data.map(record => ({
        x: record[params[0]],
        y: record[params[1]],
    }));

    const _points = await pointDataSet.toArray();
    tf.util.shuffle(_points);
    _points.pop();
    
    return _points;
}

/**
 * Create tsfjs Tensor object from array. YES
 * 
 * @param shape - Shape of the tensor
 * @param arr - Input array of objects 
 * @param field - Field name we seek in object
 * @param instance - Predator instance (optional)
 * @returns Tsfjs Tensor object
 */
predator.tensorFromArray = async (shape, arr, field, instance) => {
    shape = predator.adjustTensorShape(shape, arr, instance);

    if (instance) { instance.config.neural.layers.tensorShape = shape; }

    const fetchedArray = await arr.map(val => val[field]);

    const adjustedArray = fetchedArray.slice(0, shape.reduce((a,b) => a * b ));
    const tensor = predator.makeTensor(adjustedArray, shape);
    if (instance) { instance.predCache.push(tensor); }
    return predator.normalizeTensor(tensor);
}

/**
 * Adjust tensor dimensions. YES
 * 
 * @param shape - Tensor shape
 * @param points - Reference points array
 * @param saveTo - Instance where to save modifying parameters
 * 
 * @returns Adjusted tensor shape
 */
predator.adjustTensorShape = (shape, points, saveTo) => {
    shape.forEach((value, index) => {
        if (typeof value === 'object') {
            if (saveTo) { saveTo.config.generated.adjusted = { with: value.name, using: value.param }; }
            shape[index] = value.fn(points);
        }
    });

    return shape;
}

/**
 * Save tsfjs model into local storage. YES
 * 
 * @param model - Tsfjs model reference 
 * @param modelName - Model name
 * @param config - Predator configuration
 * @returns Saving result
 */
predator.saveModel = async (model, modelName, config) => {
    localStorage.setItem(`tfmodel/${modelName}`, JSON.stringify(config));
    return await model.save(`localstorage://${modelName}`);
}

/**
 * Attempt to retrieve model training configuration. YES
 * 
 * @param modelName - Name of a model
 * @param fallback - Fallback config
 * @returns Configuration object
 */
predator.getConfig = (modelName, fallback) => {
    const data = localStorage.getItem(`tfmodel/${modelName}`);
    if (data) { console.log(`Model config ${modelName} found.`); } else { console.log(`Model config ${modelName} not found.`); }
    return (data) ? JSON.parse(data) : fallback;
}

/**
 * Get model from localstorage by name. YES
 * 
 * @param modelName - Name of the model
 * @returns Tsfjs model reference
 */
predator.getModelByName = async (modelName) => {
    const modelInfo = (await tf.io.listModels())[`localstorage://${modelName}`];
    if (modelInfo) {
        let model = await tf.loadLayersModel(`localstorage://${modelName}`);
        model.modelName = modelName;
        return model;
    }
    return null;
}

/**
 * Retrieve model from model data object. YES
 * 
 * @param modelData - Object containing model information
 * @param noModelCallback - Function to execute if model was not found
 */
predator.unpackModel = async (modelData, noModelCallback) => {
    if (!modelData.model) {
        const model = await predator.getModelByName(modelData.name || modelData);
        if (!model) { if (noModelCallback) { noModelCallback(); } return null; }
        else { return model; }
    } else {
        return modelData.model;
    }
}

/**
 * Plot data to scatter plot. YES
 * 
 * @param values - Array of values to plot
 * @param series - Array of series to apply
 * @param modelData - Object containing model info
 * @param instance - Predator instance (optional)
 */
predator.genericPlot = (values, series, modelData, instance) => {
    modelName = (typeof modelData === 'string') ? modelData : (modelData.name || 'default');

    const config = predator.getConfig(modelName, instance.config);
    const name = `${config.system.params[0]} and ${config.system.params[1]} correlation (${modelName})`

    tfvis.render.scatterplot(
        { name },
        { values, series },
        { xLabel: 'Square feet', yLabel: 'Price' }
    );
}

/**
 * Get tensor creating function based on
 * defined tensor shape.
 * 
 * @param points - Input points
 * @param shape - Tensor shape
 * @returns Tensor creating function
 */
predator.makeTensor = (points, shape) => {
    const builder = tf[`tensor${shape.length}d`];
    return builder(points, shape);
}

/**
 * Generate dense layers based on tensor shape. YES
 * 
 * @param params - Parameters defining dense layers
 * @param tensorShape - Shape of input tensor data
 * @returns Array of dense layers
 */
predator.denseGenerator = ({ amount, units, bias, activation }, tensorShape) => {
    let layers = [];
    
    let shape = tensorShape.slice(1, -1);
    shape.push(units);

    for (let i = 0; i < amount; i++) {
        if (i === 0) {
            layers.push(
                { units, useBias: bias, inputShape: tensorShape.slice(1) }
            );
        } else if (i === amount - 1) {
            layers.push(
                { units: tensorShape.slice(-1)[0], useBias: bias, activation: activation, inputShape: shape }
            );
        } else {
            layers.push(
                { units, useBias: bias, inputShape: shape }
            );
        }
    }

    return layers;
}