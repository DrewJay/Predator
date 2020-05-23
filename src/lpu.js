/**
 * Language Processing Unit offers methods to convert fluid full text to
 * object-like data structures. It parses and parametrizes input sentences
 * and analyzes semantical points of interest.
 */
const LPU = function() {
    /**
     * Main method that attempts to parse paragraph and
     * generate configuration file from fluid paragraphs.
     * 
     * @param paragraph - Paragraph of sentences
     * @returns Tuples of parsed data
     */
    this.parse = (paragraph) => {
        const chunks = LPU.makeWordGroups(paragraph.split(' '));
        const resultTuple = [];

        for (let i = 0; i < chunks.length; i++) {
            let verbType;
            let chainOffset;
            const words = [chunks[i], chunks.slice(i, i + 2).join(' '), chunks.slice(i, i + 3).join(' ')];

            for (let q = 0; q < words.length; q++) {
                verbType = LPU.determineVerbType(words[q]);
                if (verbType) {
                    chainOffset = q;
                    break;
                }
            }

            switch (verbType) {
                case 'centric':
                    resultTuple.push([chunks[i - 1], chunks[i + chainOffset + 1]]);
                break;
                case 'leading':
                    resultTuple.push([chunks[i + chainOffset + 1], chunks[i + chainOffset + 2]]);
                break;
                case 'symbolic':
                    resultTuple.push([chunks[i - 1], chunks[i + chainOffset + 1]]);
                break;
                default: break;
            }
        }
        return resultTuple;
    }
};

/**
 * Object storing important words and phrases on which
 * parsing will be based.
 */
LPU.vocabulary = {
    verbs: {
        leading: ['will have', 'could have', 'would have', 'could use', 'would use', 'could be using'],
        centric: ['is', 'will be', 'would be', 'could be', 'is named', 'to'],
        symbolic: ['=', ':', '<>'],
    },
    compounds: [
        // System names.
        'activation function',
        'neural network',
        'system configuration',

        // Property names.
        'visual property',
        'params property',
        'csvpath property',

        // Misc names.
        'as follows',
    ],
};

/**
 * Check if word is possible compound (word consisting of
 * multiple words). Return true if yes, false if not or if
 * word is actually full compound.
 * 
 * @param word - Word to check
 * @returns True or false
 */
LPU.possibleCompound = (word) => {
    let comps = LPU.vocabulary.compounds;

    for (let comp of comps) {
        if (comp.includes(word.toLowerCase()) && comp !== word.toLowerCase()) {
            return true;
        }
    }
    return false;
};

/**
 * Group word chunks into multiple word groups based on
 * different categories.
 * 
 * @param chunks - Array of space-separated words
 * @returns New array with merged words
 */
LPU.makeWordGroups = (chunks) => {
    const groups = [];

    for (let i = 0; i < chunks.length; i++) {
        let innerIndex = i;
        let forged = chunks[i];

        while (LPU.possibleCompound(forged)) {
            forged += ` ${chunks[++innerIndex]}`;
        }

        groups.push(forged);
        i = innerIndex;
    }

    return groups;
};

/**
 * Determines if word is a verb and if it is what kind
 * of verb it is.
 * 
 * @param word - Input word
 * @returns Verb type or undefined
 */
LPU.determineVerbType = (word) => {
    let keys = Object.keys(LPU.vocabulary.verbs);

    for (let key of keys) {
        if (LPU.vocabulary.verbs[key].includes(word)) {
            return key;
        }
    }
    return undefined;
};

/**
 * Find subject of parsing in configuration objects,
 * which means keys of which value is not an object.
 * 
 * @param data - Input data object
 * @returns Array of subjects
 */
LPU.semanticPOI = (data) => {
    const keys = Object.keys(data);
    let result = [];

    for (let key of keys) {
        if (typeof data[key] === 'object') {
            result.push(LPU.semanticPOI(data[key]));
        } else {
            result.push(key);
        }
    }

    return result.flat();
};

/**
 * Primary method utilizing LPU parsing capabilities
 * and generating configuration from words.
 *
 * @param description - Describe your neural network
 * @param injectOnFailure - When parsing values fails, store the information
 *                          configuration file itself.
 * @param consumer - Configuration consumer.
 * @returns Usable configuration object or result of consuming
 */
const useWords = (description, consumer, injectOnFailure = true) => {
    const lpu = new LPU();
    const tuples = lpu.parse(description);

    // Currently symplified version, only supports system configuration layer.
    let resultConfiguration = { system: {} };
    let evidenceIndex = -1;

    for (let tuple of tuples) {
        evidenceIndex += 1;

        let key = null;
        let value = null;

        if (tuple[0].includes('property')) {
            try {
                key = tuple[0].split(' ')[0];
                value = tuple[1];

                resultConfiguration.system[key] = value;
            } catch (ConfigurationCraftingException) {
                if (injectOnFailure) {
                    resultConfiguration[`lpu::craftingexception::${evidenceIndex}`] = tuple;
                }
                continue;
            }
        }
    }

    // Either construct consumer or return configuration itself.
    if (consumer) {
        return new consumer(resultConfiguration);
    } else {
        return resultConfiguration;
    }
};
