/**
 * Pred(ictor)ator class is construct simplifying
 * all kinds of regressions and neural network model
 * creation. Based on tensorflow js.
 */
const predator = function(config) {

    // << Launch configuration cheat sheet >>
    // -> neural/model:
    //      + epochs {number} @ 10
    //      + loss {string} @ 'meanSquaredError'
    //      + optimizer {string} 'adam'
    //      + ttSplit {number} @ 2
    // -> neural/layers:
    //      + bias {boolean} @ true
    //      + activation {string} @ 'sigmoid'
    //      + amount {number} @ 3
    //      + nodes {number} @ 10
    //      + tensorShapes {array}{array} @ [[max(1), len(param[0])], [max(1)], len(param[1])]
    // -> system:
    //      + visual {boolean} @ false
    //      + params {array}{array|string}
    //      + csvPath {string}
    this.config = predator.applyDefaults(config);

    this.config.generated = {};
    this.predCache = [];
    this.points = [];

    /**
     * Combine multiple generic plots.
     * 
     * @param modelData - Object containing model name or model itself
     * @param shouldAggregate - If should simulate synthetic state
     * @param shouldPredict - If should render prediction line
     */
    this.mergePlot = async (modelData, shouldAggregate, shouldPredict) => {

        if (!this.config.system.visual) { return false; }
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
     * Generate prediction points that form a line.
     * 
     * @param modelData - Object containing model name or model itself
     * @returns Array of points
     */
    this.generatePredictionPoints = async (modelData) => {
        const model = await predator.unpackModel(modelData, modelData.noModelCallback);
        const targetDimension = this.config.neural.layers.tensorShapes[0].slice(1);
        const dimensionProduct = targetDimension.reduce((a, b) => a * b);
        const scaler = 100;
        const pointAmount = dimensionProduct * scaler;

        if (!model) { return []; }

        const [xs, ys] = tf.tidy(() => {
            const normalizedXs = tf.linspace(0, 1, pointAmount),
                  normalizedYs = model.predict(normalizedXs.reshape([scaler, targetDimension].flat()));

            const dnxs = predator.denormalizeTensor(normalizedXs, this.predCache[0]);
                  dnys = predator.denormalizeTensor(normalizedYs, this.predCache[1]);

            return [ dnxs.dataSync(), dnys.dataSync() ];
        });

        return Array.from(xs).map((val, index) => {
            return { x: val, y: ys[index] };
        });
    }

    /**
     * Attempt to make a prediction.
     * 
     * @param values - Feature values
     * @param modelData - Object containing model name or model itself
     * @returns Predicted x value
     */
    this.predict = async (values, modelData) => {
        const model = await predator.unpackModel(modelData, modelData.noModelCallback);
        if (!model) { return null; }
        await this.aggregateState(model.modelName, this);

        const paramLen = Array.isArray(this.config.system.params[0]) ? this.config.system.params[0].length : 1;
        if (paramLen !== values.length) {
            predator.error(
                `badinput`,
                `Model "${model.modelName}" expects ${paramLen} inputs but got ${values.length}.`
            );
        }

        // Except the first dimension (acting as a null), the rest of dimensions have to match the training tensor.
        const inputTensor = predator.normalizeTensor(predator.makeTensor(values, [1, this.config.neural.layers.tensorShapes[0].slice(1)].flat()), this.predCache[0]);
        const outputTensor = predator.denormalizeTensor(model.predict(inputTensor), this.predCache[1]);
        return outputTensor.dataSync();
    }

    /**
     * Aggregate particular training state so
     * we can use features it produces in process.
     * 
     * @param modelName - If used, state is aggregated from saved model.
     */
    this.aggregateState = async (modelName) => {
        this.config = predator.getConfig(modelName);
        if (!Array.isArray(this.config.neural.layers.tensorShapes[0])) { this.config.neural.layers.tensorShapes = [this.config.neural.layers.tensorShapes, this.config.neural.layers.tensorShapes]; }
        const params = this.config.system.params;
        this.points = await predator.consumeCSV(this.config.system.csvPath, params);
        this.predCache = [];
        await predator.tensorFromArray(this.config.neural.layers.tensorShapes[0], this.points, 'x', this),
        await predator.tensorFromArray(this.config.neural.layers.tensorShapes[1], this.points, 'y', this);
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
        const featureTensor = await predator.tensorFromArray(this.config.neural.layers.tensorShapes[0], this.points, 'x', this),
              labelTensor = await predator.tensorFromArray(this.config.neural.layers.tensorShapes[1], this.points, 'y', this);

        // Test-train split.
        const [trainFeatureTensor, testFeatureTensor] = tf.split(featureTensor, this.config.neural.model.ttSplit),
              [trainLabelTensor, testLabelTensor] = tf.split(labelTensor, this.config.neural.model.ttSplit);
        
        // Create tsfjs model and train it.
        const layers = predator.symmetricDNNGenerator({ amount: this.config.neural.layers.amount, units: this.config.neural.layers.nodes, bias: this.config.neural.layers.bias, activation: this.config.neural.layers.activation }, this.config.neural.layers.tensorShapes );
        this.config.generated.layers = layers;

        const model = predator.createModel(
            layers, this.config.neural.model.optimizer, this.config.neural.model.loss
        );

        const trainResult = await predator.train(model, this.config.neural.model.epochs, { trainFeatureTensor, trainLabelTensor }, this.config.system.visual);
        
        // If name is set, save the model.
        if (name) {
            await predator.saveModel(model, name, this.config);
        }

        // Calculate test loss.
        const lossTensor = model.evaluate(testFeatureTensor, testLabelTensor);
        const testLoss = await lossTensor.dataSync();
        
        // Plot the results.
        await this.mergePlot({ model: model, name }, false, true, pred);

        if (this.config.system.visual) {
            tfvis.render.barchart({ name: 'Test vs Train' }, [{ index: 'Train', value: trainResult.history.loss[this.config.neural.model.epochs - 1] }, { index: 'Test', value: testLoss }]);
        }

        return model;
    }
}

/**
 * Apply default values to missing configurations.
 */
predator.applyDefaults = (config) => {
    let neural = config.neural;
    const params = config.system.params;
    const keys = ['model/epochs', 'model/loss', 'model/optimizer', 'model/ttSplit', 'layers/bias', 'layers/activation', 'layers/amount', 'layers/nodes', 'layers/tensorShapes'];
    const defaults = [10, 'meanSquaredError', 'adam', 2, true, 'sigmoid', 3, 10, [[predator.max(1), predator.paramLength(params[0])], [predator.max(1), predator.paramLength(params[1])]]];

    if (!neural) { config.neural = {}; neural = {}; }
    if (!neural.model) { config.neural.model = {}; }
    if (!neural.layers) { config.neural.layers = {}; }

    if (!neural.model && !neural.layers) { 
        console.warn('Using default preset for standard regression task. Feel free to specify your configuration @ instance.config.'); 
    }
    
    keys.forEach((value, idx) => {
        [space, key] = value.split('/');
        if (!config.neural[space][key]) {
            config.neural[space][key] = defaults[idx];
        }
    });

    return config;
}

/**
 * Create new predator error.
 * 
 * @param name - Error name
 * @param text - Error text
 * @returns New error
 */
predator.error = (name, text) => {
    let err = new Error(text);
    err.name = `pred::${name}`;
    throw err;
}

/**
 * Get length of params, wheter it's array or a string.
 * 
 * @param param - Subject param
 * @returns Length
 */
predator.paramLength = (param) => {
    if (Array.isArray(param)) {
        return param.length;
    } else {
        return 1;
    }
}

/**
 * Return length of an input.
 * 
 * @param divide - Divisor
 * @returns Object containing operation information
 */
predator.max = (divide = 1) => {
    return {
        fn: (input) => input.length / divide,
        name: 'max',
        param: divide,
    }
}

/**
 * Normalize tensor values (downscaling).
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
 * Denormalize tensor values (upscaling).
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
 * Create tsfjs model used for training and testing.
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
 * Engage model training phase.
 * 
 * @param model - Tsfjs model reference
 * @param epochs - Amount of epochs
 * @param tensors - Feature and label tensors
 * @param showProgess - Visually show training progress
 * @returns Training data
 */
predator.train = async (model, epochs, { trainFeatureTensor, trainLabelTensor }, showProgress = false) => {
    
    let callbacks = {};

    if (showProgress) {
        const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
            { name: "Training Performance" },
            ['loss']
        );
        callbacks = { onEpochEnd };
    }
    
    return await model.fit(trainFeatureTensor, trainLabelTensor, {
        epochs,
        callbacks,
    });
}

/**
 * Read CSV from gived URL.
 * 
 * @param url - Path to CSV
 * @param params - If present, override local params
 * @returns Array-ized CSV data
 */
predator.consumeCSV = async (url, params) => {
    const data = tf.data.csv(url);
    
    const pointDataSet = await data.map(record => ({
        x: predator.spreadRecordFields(record, params[0]),
        y: predator.spreadRecordFields(record, params[1]),
    }));

    const _points = await pointDataSet.toArray();
    tf.util.shuffle(_points);
    _points.pop();

    return _points;
}

/**
 * Spread CSV record fields to form single array
 * with keys and values.
 * 
 * @param record - CSV row
 * @param params - Searched parameters
 * @returns Array of values
 */
predator.spreadRecordFields = (record, params) => {
    let values = [];

    if (!Array.isArray(params)) { return record[params]; }

    params.forEach((val) => {
        values.push(record[val]);
    });

    return values;
}

/**
 * Create tsfjs Tensor object from array.
 * 
 * @param shape - Shape of the tensor
 * @param arr - Input array of objects 
 * @param field - Field name we seek in object
 * @param instance - Predator instance (optional)
 * @returns Tsfjs Tensor object
 */
predator.tensorFromArray = async (shape, arr, field, instance) => {
    instance.shapeIndex = (field === 'x') ? 0 : 1;
    
    shape = predator.adjusttensorShapes(shape, arr, instance);

    if (instance) { instance.config.neural.layers.tensorShapes[instance.shapeIndex] = shape; }

    const fetchedArray = await arr.map(val => val[field]);

    const adjustedArray = fetchedArray.slice(0, shape.reduce((a,b) => a * b ));
    const tensor = predator.makeTensor(adjustedArray, shape);
    if (instance) { instance.predCache.push(tensor); }
    return predator.normalizeTensor(tensor);
}

/**
 * Adjust tensor dimensions.
 * 
 * @param shape - Tensor shape
 * @param points - Reference points array
 * @param saveTo - Instance where to save modifying parameters
 * @returns Adjusted tensor shape
 */
predator.adjusttensorShapes = (shape, points, saveTo) => {
    saveTo.config.generated.adjusted = saveTo.config.generated.adjusted || [];
    shape.forEach((value, index) => {
        if (typeof value === 'object') {
            if (saveTo) { saveTo.config.generated.adjusted[saveTo.shapeIndex] = { with: value.name, using: value.param }; delete saveTo.shapeindex; }
            shape[index] = value.fn(points);
        }
    });

    return shape;
}

/**
 * Save tsfjs model into local storage.
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
 * Attempt to retrieve model training configuration.
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
 * Get model from localstorage by name.
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
 * Retrieve model from model data object.
 * 
 * @param modelData - Object containing model information
 * @param noModelCallback - Function to execute if model was not found
 * @returns Tensorflow model
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
 * Plot data to scatter plot.
 * 
 * @param values - Array of values to plot
 * @param series - Array of series to apply
 * @param modelData - Object containing model info
 * @param instance - Predator instance (optional)
 */
predator.genericPlot = (values, series, modelData, instance) => {
    const modelName = (typeof modelData === 'string') ? modelData : (modelData.name || 'default');
    let featureName, labelName;

    if (instance) {
        const config = predator.getConfig(modelName, instance.config);
        featureName = config.system.params[0];
        labelName = config.system.params[1];
    } else {
        feature = 'unknown'; label = 'unknown';
    }

    const name = `${featureName} and ${labelName} correlation (${modelName})`

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
 * If tensor shape was defined as for example 3d (e.g [9, 3, 2]), it means that
 * corresponding param shape has to be three dimensional as well.
 * 
 * If input data tensor is defined as for example 3d (e.g [9, 3, 2]), results tensor has to be
 * 3d as well, with matching leftover dimensions ([, 3, 2]).
 * 
 * @param points - Input points
 * @param shape - Tensor shape
 * @returns Tensor creating function
 * 
 */
predator.makeTensor = (points, shape) => {
    const builder = tf[`tensor${shape.length}d`];
    return builder(points.flat(), shape);
}

/**
 * Generate dense layers based on tensor shape.
 * 
 * @param params - Parameters defining dense layers
 * @param tensorShapes - Shape of input tensor data
 * @returns Array of dense layers
 */
predator.symmetricDNNGenerator = ({ amount, units, bias, activation }, tensorShapes) => {
    let layers = [];
    
    let shape = tensorShapes[0].slice(1, -1);
    shape.push(units);

    for (let i = 0; i < amount; i++) {
        if (i === 0) {
            layers.push(
                { units, useBias: bias, inputShape: tensorShapes[0].slice(1) }
            );
        } else if (i === amount - 1) {
            layers.push(
                { units: tensorShapes[1].slice(-1)[0], useBias: bias, activation: activation, inputShape: shape }
            );
        } else {
            layers.push(
                { units, useBias: bias, inputShape: shape }
            );
        }
    }

    return layers;
}