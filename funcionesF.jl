using FileIO;
using DelimitedFiles;
using Statistics;
using Printf
using Images
using DelimitedFiles
using Plots
using Random
using StatsModels
using Images
using Flux, Flux.Losses
using Random:seed!
using ScikitLearn
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

# Valores numéricos a las salidas deseadas
function oneHotEncoding(feature::AbstractArray{<:Any,1},
    classes::AbstractArray{<:Any,1})
numClasses = length(classes)
numInstances = size(feature,1)
if (numClasses == 2) # 2 categorías
    boolVector = (==).(feature, classes[1]);
    return reshape(boolVector, numInstances, 1);
else # más de 2 categorías
    matrix = Array{Bool,2}(undef, length(feature), numClasses);
    for i in 1:numClasses # compara los patrones con cada categoría
        matrix[:,i] = (==).(feature, classes[i]);
    end
    return matrix;
end
end;

(oneHotEncoding)(feature::AbstractArray{<:Any,1}) =
oneHotEncoding(feature,unique(feature));

(oneHotEncoding)(feature::AbstractArray{Bool,1}) =
reshape(feature, size(feature,1), 1);


# Normalización con máximo y mínimo: intervalo [0,1]
function calculateMinMaxNormalizationParameters(val::AbstractArray{<:Real,2})
    return (maximum(val, dims=1), minimum(val, dims=1));
end;

function normalizeMinMax!(val::AbstractArray{<:Real,2},
        params::NTuple{2,AbstractArray{<:Real,2}})
    norm(v, max, min) = (max - min != 0) ? (v - min) / (max - min) : 0;
    val[:] = norm.(val, params[1], params[2]);
end;

(normalizeMinMax!)(val::AbstractArray{<:Real,2}) =
    normalizeMinMax!(val, calculateMinMaxNormalizationParameters(val));

function normalizeMinMax(val::AbstractArray{<:Real,2},
        params::NTuple{2,AbstractArray{<:Real,2}})
    return normalizeMinMax!(copy(val), params);
end;

(normalizeMinMax)(val::AbstractArray{<:Real,2}) =
    normalizeMinMax!(copy(val));

# Normalización con media y desviación típica
function calculateZeroMeanNormalizationParameters(val::AbstractArray{<:Real,2})
    return (mean(val, dims=1), std(val, dims=1));
end;

function normalizeZeroMean!(val::AbstractArray{<:Real,2},
        params::NTuple{2,AbstractArray{<:Real,2}})
    norm(v, m, d) = (d != 0) ? (v - m) / d : 0;
    val[:] = norm.(val, params[1], params[2]);
end;

(normalizeZeroMean!)(val::AbstractArray{<:Real,2}) =
    normalizeZeroMean!(val, calculateZeroMeanNormalizationParameters(val));

function normalizeZeroMean(val::AbstractArray{<:Real,2},
        params::NTuple{2,AbstractArray{<:Real,2}})
    return normalizeZeroMean!(copy(val), params);
end;

(normalizeZeroMean)(val::AbstractArray{<:Real,2}) =
    normalizeZeroMean!(copy(val));

# Repartir patrones en entrenamiento, validación y test
function holdOut(N::Int, P::Real)
    indices = randperm(N)
    test = convert(Int, round(N*P))
    return (indices[1:test], indices[test+1:end])
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    test,rest = holdOut(N,Ptest)
    val,train = holdOut(length(rest),Pval)
    return (test, val, train)
end

function crossvalidation(N::Int64, k::Int64)
    v1 = convert(Array,1:k)
    reps = convert(Int,ceil(N/length(v1))) #nº de veces que cabe v1 en N
    v2 = repeat(v1,reps)[1:N] #v2[i] indica a qué subconjunto va el patrón i
    return shuffle!(v2) #desordena el vector para que los subconjuntos sean aleatorios
end

# validación cruzada estratificada
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indices = convert(Array,1:size(targets,1))
    if (size(targets,2) == 1)
        class1 = (!iszero).(indices)
        class2 = iszero.(indices)
        indices[class1] = crossvalidation(sum(class1),k)
        indices[class2] = crossvalidation(sum(class2),k)
    else
        for class in eachcol(targets) #reparte aleatoriamente los patrones de cada clase
            indices[class] = crossvalidation(sum(class),k)
        end
    end
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = oneHotEncoding(targets,unique(targets))
    crossvalidation(classes,k)
end

# Construir RNA
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1},
        numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=
        fill(σ, length(topology)))
    ann = Chain();
    numInputsLayer = numInputs;
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ));
        numInputsLayer = numOutputsLayer;
    end;
    if (numClasses <= 2)
        ann = Chain(ann...,Dense(numInputsLayer,1,σ));
    else
        ann = Chain(ann...,Dense(numInputsLayer,numOutputs,identity),softmax);
    end;
end;


# Clasificar salidas (pertenencia a clases)
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if (size(outputs,2) == 1)
        return (>=).(outputs, threshold);
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs;
    end;
end;

# Precisión del modelo
function accuracy(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    mean(targets .== outputs);
end;

(accuracy)(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) =
    if (size(outputs,2) <= 2)
        return accuracy(outputs[:,1], targets[:,1])
    else
        classComparison = targets .== outputs;
        correctClassifications = all(classComparison, dims=2);
        return mean(correctClassifications);
    end;

function accuracy(outputs::AbstractArray{<:Real,1},
        targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = classifyOutputs(reshape(outputs,size(outputs,1),1); threshold);
    return accuracy(convert(Array{Bool,1}, outputs[:,1]), targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2},
        targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    if (size(outputs,2) == 1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        outputs = classifyOutputs(outputs; threshold);
        return accuracy(outputs, targets);
    end;
end;

# Métricas
function confusionMatrix(outputs::AbstractArray{Bool,1},
        targets::AbstractArray{Bool,1})
    v = (==).(outputs,targets)
    f = (!=).(outputs,targets)
    vp = sum((!iszero).(outputs[v]))
    vn = sum(iszero.(outputs[v]))
    fp = sum((!iszero).(outputs[f]))
    fn = sum(iszero.(outputs[f]))
    if (vp + fn + fp != 0)
        sensitivity = fn+vp != 0 ? vp/(fn+vp) : 0
        positivePredVal = vp+fp != 0 ? vp/(vp+fp) : 0
    else
        sensitivity = 1
        positivePredVal = 1
    end
    if (vn + fn + fp != 0)
        specificity = fp+vn != 0 ? vn/(fp+vn) : 0
        negativePredVal = vn+fn != 0 ? vn/(vn+fn) : 0
    else
        specificity = 1
        negativePredVal = 1
    end
    if (vn + vp + fn + fp != 0)
        accuracy = (vn + vp)/(vn + vp + fn + fp)
        errorRate = (fn + fp)/(vn + vp + fn + fp)
    else
        accuracy = 0
        errorRate = 0
    end
    f1score = sensitivity+positivePredVal != 0 ?
        (2*sensitivity*positivePredVal)/(sensitivity+positivePredVal) : 0
    matrix = [vn fp; fn vp]
    return (accuracy,errorRate,sensitivity,specificity,
            positivePredVal,negativePredVal,f1score,matrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},
        targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    boolOutputs = (x -> x < 0.5 ? false : true).(outputs)
    confusionMatrix(boolOutputs,targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2},
        targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert (size(outputs,2) == size(targets,2)) "outputs y targets no tienen el mismo número de columnas"
    @assert (size(outputs,2) != 2) "outputs no puede tener 2 clases"
    @assert (size(targets,2) != 2) "targets no puede tener 2 clases"

    if (size(outputs,2) == 1)
        confusionMatrix(outputs[:,1],targets[:,1])
    else
        sensitivity = zeros(numClasses)
        specificity = zeros(numClasses)
        vpp = zeros(numClasses)
        vpn = zeros(numClasses)
        f1 = zeros(numClasses)

        for class in 1:numClasses
            v = confusionMatrix(outputs[:,class],targets[:,class])
            sensitivity[class] = v[3]
            specificity[class] = v[4]
            vpp[class] = v[5]
            vpn[class] = v[6]
            f1[class] = v[7]
        end

        matrix = zeros(Int,numClasses,numClasses)
        for i in 1:numClasses
            for j in 1:numClasses
                classified(x,y) = x==true && y == true
                matrix[i,j] = sum((classified).(targets[:,i],outputs[:,j]))
            end
        end

        globalSensitivity,globalSpecificity,globalVpp,globalVpn,globalF1 = 0,0,0,0,0
        instances = size(targets,1)
        if weighted == true
            for class in 1:numClasses
                classInstances = sum(targets[:,class]) / instances
                globalSensitivity += sensitivity[class] * classInstances
                globalSpecificity += specificity[class] * classInstances
                globalVpp += vpp[class] * classInstances
                globalVpn += vpn[class] * classInstances
                globalF1 += f1[class] * classInstances
            end
        else
            globalSensitivity += sum(sensitivity)
            globalSpecificity += sum(specificity)
            globalVpp += sum(vpp)
            globalVpn += sum(vpn)
            globalF1 += sum(f1)
        end
        globalSensitivity,globalSpecificity,globalVpp,globalVpn,globalF1 ./ instances
        acc = accuracy(outputs,targets)
        errorRate = 1 - acc
        return (acc,errorRate,globalSensitivity,globalSpecificity,
                globalVpp,globalVpn,globalF1,matrix)
    end
end

function confusionMatrix(outputs::AbstractArray{<:Real,2},
        targets::AbstractArray{Bool,2}; weighted::Bool=true)
    confusionMatrix(classifyOutputs(outputs),targets)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1},
        targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs])) # para cada salida, comprueba que sea una de las categorías válidas
    classes = unique(targets)
    outputs = oneHotEncoding(outputs,classes)
    targets = oneHotEncoding(targets,classes)
    confusionMatrix(outputs,targets)
end

function oneVSall(model, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
    outputs = Array{Float32,2}(undef, numInstances, numClasses);
    for numClass in 1:numClasses
        newModel = deepcopy(model);
        StatsModels.fit!(newModel, inputs', targets[:,numClass]');
        outputs[:,numClass] .= newModel(inputs);
    end;
    outputs = softmax(outputs')';
    vmax = maximum(outputs, dims=2);
    outputs = (outputs .== vmax);
end

function printMatrix(v, n)
    println("Matriz de confusión: ")
    if (n == 2)
        println("              ┌─────────────────┐")
        println("              │    Predicción   │")
        println("              ├────────┬────────┤")
        println("              │Negativo│Positivo│")
        println("┌────┬────────┼────────┼────────┤")
        print("│    │Negativo│  "); Printf,@printf("%.2f  │  %.2f  │\n",v[1,1],v[1,2])
        println("│Real├────────┼────────┼────────┤")
        print("│    │Positivo│  "); Printf,@printf("%.2f  │  %.2f  │\n",v[2,1],v[2,2])
        println("└────┴────────┴────────┴────────┘")
    else
        print("            ")
        print("┌")
        for i in 1:n-1
            print("───────────┬")
        end
        println("───────────┐")
        print("            ")
        print("│")
        for i in 1:n
            print(" Clase "*string(i)*"   │")
        end
        println("")
        print("┌───────────┼")
        for i in 1:n-1
            print("───────────┼")
        end
        print("───────────┤")
        println("")
        for i in 1:n-1
            print("│ Clase "*string(i)*"   │")
            for j in 1:n
                Printf,@printf("   %.2f    │",v[i,j])
            end
            println("")
            print("├───────────┼")
            for j in 1:n-1
                print("───────────┼")
            end
            print("───────────┤")
            println("")
        end
        print("│ Clase "*string(n)*"   │")
        for j in 1:n
            Printf,@printf("   %.2f    │",v[n,j])
        end
        println("")
        print("└───────────┴")
        for j in 1:n-1
            print("───────────┴")
        end
        print("───────────┘")
        println("")
    end
end

function printMetrics(v::Tuple, numClasses)
    println("Valor de precisión: ",v[1])
    println("Tasa de fallo: ",v[2])
    println("Sensibilidad: ",v[3])
    println("Especificidad: ",v[4])
    println("Valor predictivo positivo: ",v[5])
    println("Valor predictivo negativo: ",v[6])
    println("F1-score: ",v[7])
    println("Matriz de confusión: ")
    printMatrix(v[8], numClasses)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1},
        targets::AbstractArray{Bool,1})
    v = confusionMatrix(outputs,targets)
    printMetrics(v)
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
        targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    v = confusionMatrix(outputs,targets;threshold)
    printMetrics(v)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,2},
        targets::AbstractArray{Bool,2}; weighted::Bool=true)
    v = confusionMatrix(outputs,targets;weighted)
    printMetrics(v)
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
        targets::AbstractArray{Bool,2}; weighted::Bool=true)
    v = confusionMatrix(outputs,targets;weighted)
    printMetrics(v)
end



function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    (inputs, targets) = dataset;

    @assert(size(inputs,1)==size(targets,1));


    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    trainingLosses = Float32[];

    numEpoch = 0;
    
    trainingLoss = loss(inputs', targets');
    
    push!(trainingLosses, trainingLoss);
    
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        
        Flux.train!(loss, Flux.params(ann), [(inputs', targets')], ADAM(learningRate));

        
        numEpoch += 1;
        
        trainingLoss = loss(inputs', targets');
        
        push!(trainingLosses, trainingLoss);
        
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    end;

    
    return (ann, trainingLosses);
end;


trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) = trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    
    @assert(size(trainingInputs,   1)==size(trainingTargets,   1));
    @assert(size(testInputs,       1)==size(testTargets,       1));
    @assert(size(validationInputs, 1)==size(validationTargets, 1));
    
    !isempty(validationInputs)  && @assert(size(trainingInputs, 2)==size(validationInputs, 2));
    !isempty(validationTargets) && @assert(size(trainingTargets,2)==size(validationTargets,2));
    
    !isempty(testInputs)  && @assert(size(trainingInputs, 2)==size(testInputs, 2));
    !isempty(testTargets) && @assert(size(trainingTargets,2)==size(testTargets,2));

    
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions);
    
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    
    trainingLosses   = Float32[];
    validationLosses = Float32[];
    testLosses       = Float32[];

    
    numEpoch = 0;

   
    function calculateLossValues()
        trainingLoss = loss(trainingInputs', trainingTargets');
        showText && print("Epoch ", numEpoch, ": Training loss: ", trainingLoss);
        push!(trainingLosses, trainingLoss);
        if !isempty(validationInputs)
            validationLoss = loss(validationInputs', validationTargets');
            showText && print(" - validation loss: ", validationLoss);
            push!(validationLosses, validationLoss);
        else
            validationLoss = NaN;
        end;
        if !isempty(testInputs)
            testLoss       = loss(testInputs', testTargets');
            showText && print(" - test loss: ", testLoss);
            push!(testLosses, testLoss);
        else
            testLoss = NaN;
        end;
        showText && println("");
        return (trainingLoss, validationLoss, testLoss);
    end;

    (trainingLoss, validationLoss, _) = calculateLossValues();

    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    bestANN = deepcopy(ann);

    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)

        Flux.train!(loss, Flux.params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));

        numEpoch += 1;

        (trainingLoss, validationLoss, _) = calculateLossValues();

        if (!isempty(validationInputs))
            if (validationLoss<bestValidationLoss)
                bestValidationLoss = validationLoss;
                numEpochsValidation = 0;
                bestANN = deepcopy(ann);
            else
                numEpochsValidation += 1;
            end;
        end;

    end;

    if isempty(validationInputs)
        bestANN = ann;
    end;

    return (bestANN, trainingLosses, validationLosses, testLosses);
end;


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)); validationDataset=(validationInputs, reshape(validationTargets, length(validationTargets), 1)), testDataset=(testInputs, reshape(testTargets, length(testTargets), 1)), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showText);
end;


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    kFoldIndices::     Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)

    numFolds = maximum(kFoldIndices);

    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    for numFold in 1:numFolds

        trainingInputs    = inputs[kFoldIndices.!=numFold,:];
        testInputs        = inputs[kFoldIndices.==numFold,:];
        trainingTargets   = targets[kFoldIndices.!=numFold,:];
        testTargets       = targets[kFoldIndices.==numFold,:];

        testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsANNTraining);
        testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsANNTraining);

        for numTraining in 1:numRepetitionsANNTraining

            if validationRatio>0

                (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));

                ann, = trainClassANN(topology, (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

            else

                ann, = trainClassANN(topology, (trainingInputs, trainingTargets),
                    testDataset = (testInputs,     testTargets);
                    maxEpochs=numMaxEpochs, learningRate=learningRate);

            end;

            (acc, _, _, _, _, _, F1, _) = confusionMatrix(ann(testInputs')', testTargets);

            testAccuraciesEachRepetition[numTraining] = acc;
            testF1EachRepetition[numTraining]         = F1;

        end;

        testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
        testF1[numFold]         = mean(testF1EachRepetition);

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end;

    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

end;


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    kFoldIndices::     Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)

    (trainingInputs,   trainingTargets)   = trainingDataset;

    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)), kFoldIndices; transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

end;


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})

    @assert(size(inputs,1)==length(targets));

    classes = unique(targets);

    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);
    testErrorRate = Array{Float64,1}(undef, numFolds);
    testRecall         = Array{Float64,1}(undef, numFolds);
    testSpecificity = Array{Float64,1}(undef, numFolds);
    testPrecision         = Array{Float64,1}(undef, numFolds);
    testNPV = Array{Float64,1}(undef, numFolds);
    testConfMatrix         = Array{Array{Float64, 2},1}(undef, numFolds);

    for numFold in 1:numFolds

        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];

            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            model = fit!(model, trainingInputs, trainingTargets);

            testOutputs = predict(model, testInputs);

            (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(testOutputs, testTargets);

        else

            @assert(modelType==:ANN);

            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

        
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testErrorRateEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testRecallEachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testSpecificityEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testPrecisionEachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testNPVEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testConfMatrixEachRepetition         = Array{Array{Float64, 2},1}(undef, modelHyperparameters["numExecutions"]);

            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));

                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                        validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                        testDataset =       (testInputs,                          testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                else

                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                        testDataset = (testInputs,     testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;
                model = ann;

                (testAccuraciesEachRepetition[numTraining], testErrorRateEachRepetition[numTraining], testRecallEachRepetition[numTraining], testSpecificityEachRepetition[numTraining], testPrecisionEachRepetition[numTraining], testNPVEachRepetition[numTraining], testF1EachRepetition[numTraining], testConfMatrixEachRepetition[numTraining]) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);
            errorRate = mean(testErrorRateEachRepetition)
            recall = mean(testRecallEachRepetition)
            specificity = mean(testSpecificityEachRepetition)
            precision = mean(testPrecisionEachRepetition)
            NPV = mean(testNPVEachRepetition)
            confMatrix = mean(testConfMatrixEachRepetition)

        end;

        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;
        testErrorRate[numFold] = errorRate;
        testRecall[numFold]         = recall;
        testSpecificity[numFold] = specificity;
        testPrecision[numFold]         = precision;
        testNPV[numFold] = NPV;
        testConfMatrix[numFold]         = confMatrix;


    end; 
    return ((mean(testAccuracies),mean(testErrorRate),mean(testRecall),mean(testSpecificity),mean(testPrecision),mean(testNPV),mean(testF1),mean(testConfMatrix)));

end;