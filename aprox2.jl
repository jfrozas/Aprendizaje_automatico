include("funcionesF.jl")
numClasses = 4;

function calculateBoundingBox(imagen)
    matrizBN = Gray.(imagen); # valor de los píxeles en escala de grises (0=negro, 1=blanco)
    pixelesNegros = matrizBN .< 0.4; # diferenciar píxeles blancos y negros
    
    labelArray = ImageMorphology.label_components(pixelesNegros); # asigna un valor numérico a cada objeto de la imagen
        # cada píxel tiene el valor del objeto al que pertenece
    etiquetasEliminar = findall(ImageMorphology.component_lengths(labelArray) .<= 500) .- 1; # los objetos con menos de 1000 píxeles
    pixelesNegros2 = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray]; # se eliminan
    labelArray2 = ImageMorphology.label_components(pixelesNegros2); # recalcular objetos
    
    imagenBN = RGB.(pixelesNegros2,pixelesNegros2,pixelesNegros2); # imagen solo con los objetos grandes
    imagenBox = copy(imagen); # copia la imagen original para dibujarle la bounding box sin sobreescribirla
    
    boundingBoxes = ImageMorphology.component_boxes(labelArray2)[2:end]; # calcula la bounding box de cada objeto
    ancho = 0;
    alto = 0;
    for boundingBox in boundingBoxes # dibuja las bounding box
        x1 = boundingBox[1][1];
        y1 = boundingBox[1][2];
        x2 = boundingBox[2][1];
        y2 = boundingBox[2][2];
        imagenBN[ x1:x2 , y1 ] .= RGB(0,1,0); # en la imagen en blanco y negro
        imagenBox[ x1:x2 , y1 ] .= RGB(0,1,0); # y en la imagen original
        imagenBN[ x1:x2 , y2 ] .= RGB(0,1,0);
        imagenBox[ x1:x2 , y2 ] .= RGB(0,1,0);
        imagenBN[ x1 , y1:y2 ] .= RGB(0,1,0);
        imagenBox[ x1 , y1:y2 ] .= RGB(0,1,0);
        imagenBN[ x2 , y1:y2 ] .= RGB(0,1,0);
        imagenBox[ x2 , y1:y2 ] .= RGB(0,1,0);
        alto = x2-x1;
        ancho = y2-y1;
    end;

    gradient = cgrad(:hsv);
    colores(x) = gradient[(x*11)%(size(gradient)[1])+1]; # hash para colorear los objetos (estético, no hace falta)

    mosaicview(imagen, Gray.(pixelesNegros), colores.(labelArray), Gray.(pixelesNegros2), imagenBN, imagenBox, ; nrow=1, npad=10)

    if length(boundingBoxes) == 0
        return 0
    else
        return alto/ancho;
    end
end;
# Cargar datos
numPeones = 50
numDamas = 53
numCaballos = 50
numAlfiles = 49
numPatterns = numPeones + numDamas + numAlfiles + numCaballos; # numero de fotos que tengamos disponibles
inputs = Array{Float32,2}(undef, numPatterns,1);
targets = Array{String,1}(undef, numPatterns);
invalidInputs = Array{Int16,1}(undef, 0);

total = 0

damas = Array{Any,1}(undef, numPatterns);
for i in 1:numDamas
    damas[i] = load("data/dama"*string(i)*".png")
    inputs[i] = calculateBoundingBox(damas[i])
    if (inputs[i] == 0) # en algunas fotos no calcula bien la bounding box
        push!(invalidInputs,i) # se guardan como inválidas para borrarlas
    end
    targets[total+i] = "dama"
end;

total += numDamas

peones = Array{Any,1}(undef, numPatterns);
for i in 1:numPeones
    peones[i] = load("data/peon"*string(i)*".png")
    inputs[total+i] = calculateBoundingBox(peones[i])
    if (inputs[total+i] == 0)
        push!(invalidInputs,total+i)
    end
    targets[total+i] = "peón"
end;

total += numPeones

 caballos = Array{Any,1}(undef, numPatterns);
 for i in 1:numCaballos
     caballos[i] = load("data/caballo"*string(i)*".png")
     inputs[total+i] = calculateBoundingBox(caballos[i])
     if (inputs[total+i] == 0)
         push!(invalidInputs,total+i)
     end
     targets[total+i] = "caballo"
 end;

 total += numCaballos

 alfiles = Array{Any,1}(undef, numPatterns);
 for i in 1:numAlfiles
     alfiles[i] = load("data/alfil"*string(i)*".png")
     inputs[total+i] = calculateBoundingBox(alfiles[i])
     if (inputs[total+i] == 0)
         push!(invalidInputs,total+i)
     end
     targets[total+i] = "alfil"
 end;

# Nos quedamos solo con las entradas válidas
finalInputs = inputs[setdiff(1:size(inputs,1),invalidInputs),:]
finalTargets = targets[setdiff(1:size(inputs,1),invalidInputs)]


seed!(1);

numFolds = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [[2],[8], [16], [4,2], [8,4], [8,4,2], [2,8], [16,8]];
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 400; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsANNTraining = 10; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Parametros del SVM
kernel = ["rbf", "linear", "poly", "sigmoid", "rbf", "linear", "poly", "sigmoid"];
kernelDegree = 3;
kernelGamma = 2;
C=[1, 1, 1, 1, 3, 3, 3, 3];

# Parametros del arbol de decision
maxDepth = [2, 4, 8, 10, 13, 14];

# Parapetros de kNN
numNeighbors = [2, 5, 10, 12, 15, 16];


# Creamos los indices de validacion cruzada
crossValidationIndices = crossvalidation(finalTargets, numFolds);

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(finalInputs);

#Entrenamos las RR.NN.AA.
for i in 1:8
    modelHyperparameters = Dict();
    modelHyperparameters["topology"] = topology[i];
    modelHyperparameters["learningRate"] = learningRate;
    modelHyperparameters["validationRatio"] = validationRatio;
    modelHyperparameters["numExecutions"] = numRepetitionsANNTraining;
    modelHyperparameters["maxEpochs"] = numMaxEpochs;
    modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
    println("\n\n\nANN")
    print("Topology: ")
    println(topology[i])
    print("LearningRate: ")
    println(learningRate)
    print("ValidationRatio: ")
    println(validationRatio)
    print("NumRepetitionsANNTraining: ")
    println(numRepetitionsANNTraining)
    print("NumMaxEpochs: ")
    println(numMaxEpochs)
    print("MaxEpochsVal: ")
    println(maxEpochsVal)
    println("")
    j = modelCrossValidation(:ANN, modelHyperparameters, finalInputs, finalTargets, crossValidationIndices);
    printMetrics(j, numClasses)

end

# Entrenamos las SVM
for i in 1:8
    modelHyperparameters = Dict();
    modelHyperparameters["kernel"] = kernel[i];
    modelHyperparameters["kernelDegree"] = kernelDegree;
    modelHyperparameters["kernelGamma"] = kernelGamma;
    modelHyperparameters["C"] = C[i];
    println("\n\n\nSVM")
    print("Kernel: ")
    println(kernel[i])
    print("KernelDegree: ")
    println(kernelDegree)
    print("KernelGamma: ")
    println(kernelGamma)
    print("C: ")
    println(C[i])
    println("")
    j = modelCrossValidation(:SVM, modelHyperparameters, finalInputs, finalTargets, crossValidationIndices);
    printMetrics(j, numClasses)

    
end

# Entrenamos los arboles de decision
for i in 1:6
    println("\n\n\nDecisionTree")
    print("MaxDepth: ")
    println(maxDepth[i])
    println("")
    j = modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth[i]), finalInputs, finalTargets, crossValidationIndices);
    printMetrics(j, numClasses)

end


# Entrenamos los kNN
for i in 1:6
    println("\n\n\nkNN")
    print("NumNeighbors: ")
    println(numNeighbors[i])
    println("")
    j = modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors[i]), finalInputs, finalTargets, crossValidationIndices);
    printMetrics(j, numClasses)
end