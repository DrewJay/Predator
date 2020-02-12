/**
 * Get element by ID shortform.
 * 
 * @param id - ID of element
 * @returns HTML element
 */
const gid = (id) => {
    return document.getElementById(id);
}

/**
 * Toggle loader visibility.
 */
const toggleLoad = () => {
    const opacity = gid('loader').style.opacity;

    if (opacity == 1) {
        gid('loader').style.opacity = 0;
        gid('main').style.opacity = 1;
        gid('main').style.pointerEvents = 'inherit';
    } else {
        gid('loader').style.opacity = 1;
        gid('main').style.opacity = .5;
        gid('main').style.pointerEvents = 'none';
    }
}

/**
 * Make prediction and show results.
 * 
 * @param input - Prediction input
 */
const makeprediction = async (input) => {
    const modelName = prompt('Choose a model:');
    toggleLoad();
    const arrayized = input.toString().split(',').map((val) => parseInt(val.replace(' ', '')));

    try {
        const result = await pred.predict(arrayized, { name: modelName });
        const html = result ? `For value <span class='input'>${arrayized.join('/')}</span> model <span class='name'>${modelName}</span> predicted result <span class='result'>${Math.floor(result)}</span>.` : `Model does not exist.`
        gid('display').innerHTML = html;
    } catch (error) {
        gid('display').innerHTML = `${error.name} - ${error.message}`;
        throw error;
    } finally {
        toggleLoad();
    }
}

/**
 * Train a new model.
 * 
 * @param modelName - The name of model 
 */
const domTrain = async (modelName) => {
    if (confirm(JSON.stringify(pred.config))) {
        toggleLoad();
        await pred.session(modelName);
        toggleLoad();
    }
}

/**
 * Create a list of model names.
 */
const listModels = async() => {
    const models = await tf.io.listModels();
    const names = Object.keys(models).map((name, idx) => `<span class='model-${idx}'>${name.replace('localstorage://', '')}</span>`).join(', ') || 'none';
    gid('display').innerHTML = `Available models: ${names}.`;
};

/**
 * Show graph of model.
 */
const renderModel = async() => {
    const modelName = prompt('Choose a model:');
    toggleLoad();
    await pred.mergePlot({ name: modelName, noModelCallback: () => { gid('display').innerHTML = `Model does not exist.`; } }, true, true, pred);
    toggleLoad();
}

/**
 * Dispatch general event handlers.
 */
const dispatchHandlers = () => {
    document.onclick = (e) => {
        if (e.target.className.includes('model-')) {
            showBox(e.target.innerHTML, e.target);
        }
    }

    gid('draggable').oncontextmenu = (e) => {
        e.preventDefault();
        gid('draggable').classList.remove('shown');
    }

    gid('draggable').onmousedown = (e) => {
        const originX = e.clientX,
              originY = e.clientY,
              originLeft = gid('draggable').offsetLeft,
              originTop = gid('draggable').offsetTop;

        gid('draggable').onmousemove = (_e) => {
            const newX = _e.clientX;
            const newY = _e.clientY;
            gid('draggable').style.left = `${originLeft + newX - originX}px`;
            gid('draggable').style.top = `${originTop + newY - originY}px`;
        }
    }

    gid('draggable').onmouseup = () => {gid('draggable').onmousemove = () => {};}
}

/**
 * Show model info box.
 * 
 * @param modelName - Name of model to show info about 
 * @param target - Target node of the click
 */
const showBox = (modelName, target) => {

    const _config = Predator.getConfig(modelName);

    if (!gid('draggable').className.includes('shown')) {
        gid('draggable').classList.add('shown');
        gid('draggable').style.top = `${target.offsetTop - 216.5}px`;
        gid('draggable').style.left = `${target.offsetLeft}px`;
    }

    gid('draggable').innerHTML = `
        <div class='name'>${modelName} model</div>
        <span>Epochs</span>
        <span>${_config.neural.model.epochs}</span>
        <span>Loss Function</span>
        <span>${_config.neural.model.loss}</span>
        <span>Activation Function</span>
        <span>${_config.neural.layers.activation}</span>
        <span>Optimizer</span>
        <span>${_config.neural.model.optimizer}</span>
        <span>Bias</span>
        <span>${_config.neural.layers.bias}</span>
        <span>Params</span>
        <span>${_config.system.params.join(' -> ')}</span>
        <span>Tensor Shape</span>
        <span>${_config.neural.layers.tensorShapes.join(' and ')}</span>
        <span>Layers Amount</span>
        <span>${_config.neural.layers.amount}</span>
        <span>Layer Nodes</span>
        <span>${_config.neural.layers.nodes}</span>
    `;
}