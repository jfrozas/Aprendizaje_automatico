include("funcionesF.jl")
numClasses = 6;

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


function calculateSimetriaX(imagen, valor)
    original = imagen
    pixelesNegros = Gray.(imagen) .< valor
    labelArray = ImageMorphology.label_components(pixelesNegros) # etiqueta cada objeto de la imagen
    lengths = ImageMorphology.component_lengths(labelArray) # tamaños de los objetos
    mayorObjeto = maximum(lengths[2:end])
    etiquetasEliminar = findall(lengths .!= mayorObjeto) .- 1 # los objetos con menos de 1000 píxeles
    pixelesNegros2 = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray]; # se eliminan
    labelArray = ImageMorphology.label_components(pixelesNegros2); # recalcular objetos
    boundingBox = ImageMorphology.component_boxes(labelArray)[2]; # calcula la bounding box de l 
    
    
    x1 = boundingBox[1][1]
    y1 = boundingBox[1][2]
    x2 = boundingBox[2][1]
    y2 = boundingBox[2][2]
    
    if(size(imagen[x1:x2,y1:y2]) < convert.(Int,round.(size(imagen).*0.73))) #tam box < tam imagen * 0.73
        return calculateSimetriaX(imagen, valor+0.05)
    end

    imagen = Gray.(pixelesNegros2)[x1:x2,y1:y2]
    imagen1 = imagen

    imagen = imagen[convert(Int,round(size(imagen, 1)/3)):size(imagen, 1)-convert(Int,round(size(imagen, 1)/3)),:]

    halfUp = imagen[1:convert(Int,round(size(imagen, 1)/2)), :]
    halfDown = imagen[(convert(Int,round(size(imagen, 1)/2))+1):size(imagen,1), :]

    if(size(halfUp, 1)>size(halfDown,1))
        halfUp = halfUp[1:size(halfUp,1)-1,:]
    elseif(size(halfUp, 1)<size(halfDown,1))
        halfDown = halfDown[2:size(halfDown,1),:]
    end
    

    halfUp = convert.(Float64, halfUp)
    halfDown = convert.(Float64, halfDown)
    


    med = Array{Float64, 2}(undef, size(halfUp, 1), size(halfUp,2))
    for i in 1:size(halfUp, 1)
        for j in 1 : size(halfUp,2)
            if(halfDown[size(halfUp, 1)-i+1,j] == halfUp[i,j])
                med[i,j] = 1
            else
                med[i,j] = 0
            end
        
        end
    end

    #mosaicview(original,imagen1,imagen,halfUp, halfDown, Gray.(med); nrow=1, npad=10)

    return mean(med)
end;

function calculateSimetriaY(imagen, valor)
    original = imagen
    pixelesNegros = Gray.(imagen) .< valor
    labelArray = ImageMorphology.label_components(pixelesNegros) # etiqueta cada objeto de la imagen
    lengths = ImageMorphology.component_lengths(labelArray) # tamaños de los objetos
    mayorObjeto = maximum(lengths[2:end])
    etiquetasEliminar = findall(lengths .!= mayorObjeto) .- 1 # los objetos con menos de 1000 píxeles
    pixelesNegros2 = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray]; # se eliminan
    labelArray = ImageMorphology.label_components(pixelesNegros2); # recalcular objetos
    boundingBox = ImageMorphology.component_boxes(labelArray)[2]; # calcula la bounding box de cada objeto
    

    x1 = boundingBox[1][1]
    y1 = boundingBox[1][2]
    x2 = boundingBox[2][1]
    y2 = boundingBox[2][2]

    if(size(imagen[x1:x2,y1:y2]) < convert.(Int,round.(size(imagen).*0.73))) #tam box < tam imagen * 0.73
        return calculateSimetriaY(imagen, valor+0.05)
    end

    imagen = Gray.(pixelesNegros2)[x1:x2,y1:y2]

    halfUp = imagen[:,1:convert(Int,round(size(imagen, 2)/2))]
    halfDown = imagen[:, (convert(Int,round(size(imagen, 2)/2))+1):size(imagen,2)]
    
    if(size(halfUp, 2)>size(halfDown,2))
        halfUp = halfUp[:,1:size(halfUp,2)-1]
    elseif(size(halfUp, 2)<size(halfDown,2))
        halfDown = halfDown[:,2:size(halfDown,2)]
    end
    

    halfUp = convert.(Float64, halfUp)
    halfDown = convert.(Float64, halfDown)
    


    med = Array{Float64, 2}(undef, size(halfUp, 1), size(halfUp,2))
    for i in 1:size(halfUp, 1)
        for j in 1 : size(halfUp,2)
            if(halfDown[i,size(halfUp, 2)-j+1] == halfUp[i,j])
                med[i,j] = 1
            else
                med[i,j] = 0
            end
        
        end
    end

    # med = halfDown[length(halfDown):-1:1] .== halfUp

    #mosaicview(original,imagen,halfUp, halfDown, Gray.(med); nrow=1, npad=10)

    return mean(med)

end;

function calcularContorno(imagen, valor)
    pixelesNegros = Gray.(imagen) .< valor
    labelArray = ImageMorphology.label_components(pixelesNegros) # etiqueta cada objeto de la imagen
    lengths = ImageMorphology.component_lengths(labelArray) # tamaños de los objetos
    mayorObjeto = maximum(lengths[2:end]) # lengths[1] no cuenta porque es el fondo
    etiquetasEliminar = findall(lengths .!= mayorObjeto) .- 1 # solo se deja el objeto más grande, que es la pieza
    pixelesNegros = [!in(etiqueta,etiquetasEliminar) for etiqueta in labelArray] # pone un 0 en cada píxel borrado
    labelArray = ImageMorphology.label_components(pixelesNegros); # recalcular objetos
    boundingBox = ImageMorphology.component_boxes(labelArray)[2]; # calcula la bounding box de cada objeto
    x1 = boundingBox[1][1]
    y1 = boundingBox[1][2]
    x2 = boundingBox[2][1]
    y2 = boundingBox[2][2]

    if(size(imagen[x1:x2,y1:y2]) < convert.(Int,round.(size(imagen).*0.73))) #tam box < tam imagen * 0.73
        return calcularContorno(imagen, valor+0.05)
    end

    imagen = imagen[x1:x2,y1:y2]
    pixelesNegros = pixelesNegros[x1:x2,y1:y2]

    tam = convert(Int,round(size(imagen,1)/5)) # nº de filas de cada parte

    # cuartos superior e inferior de la imagen original
    imgUp = imagen[1:tam,:]
    imgDown = imagen[end-tam:end,:]

    # matrices donde se dibujan los píxeles de contorno sobre fondo negro
    up = pixelesNegros[1:tam,:]
    down = pixelesNegros[end-tam+1:end,:]
    contornoUp = fill(RGB(0,0,0), size(up))
    contornoDown = fill(RGB(0,0,0), size(down))

    # si no encuentra pieza en una columna (valor nothing que no se puede usar después como índice)
    # dibuja el contorno en la primera fila de la imagen (luego se ignora la primera fila)
    deleteNothingUp = (x,y)::Tuple -> return x === nothing ? (size(up,1),y) : (x,y)
    deleteNothingDown = (x,y)::Tuple -> return x === nothing ? (1,y) : (x,y)

    # busca el primer píxel de pieza en una columna
    # index es el nº de columna y col la propia columna como array, que lo devuelve enumerate
    firstPiecePixel = (index,col)::Tuple -> (findfirst(!iszero,col),index)
    top = deleteNothingUp.(firstPiecePixel.(enumerate(eachcol(up))))

    # lo mismo para buscar el último píxel en cada columna del cuarto inferior
    lastPiecePixel = (index,col)::Tuple -> (findlast(!iszero,col),index)
    bottom = deleteNothingDown.(lastPiecePixel.(enumerate(eachcol(down))))

    # dibuja los píxeles de contorno sobre la imagen ya dividida y sobre el fondo negro
    for i in 1:size(imagen,2)
        contornoUp[top[i]...] = RGB(1,0,0)
        contornoDown[bottom[i]...] = RGB(1,0,0)
        imgUp[top[i]...] = RGB(1,0,0)
        imgDown[bottom[i]...] = RGB(1,0,0)
    end

    # cuenta los píxeles de contorno de cada parte
    numPixelsTop = sum((!=).(contornoUp[1:end-1,:],RGB(0,0,0))) #no tiene en cuenta la última fila, que es donde están los nulos
    numPixelsDown = sum((!=).(contornoDown[2:end,:],RGB(0,0,0))) # aquí los nulos están en la primera

    # dibuja ignorando la primera fila
    #mosaicview(contornoUp[1:end-1,:], contornoDown[2:end,:], imgUp[1:end-1,:], imgDown[2:end,:]; nrow=2, npad=10)

    return numPixelsDown / numPixelsTop
end

function calculateAnchoMedio(imagen, valor)
    pixelesNegros = Gray.(imagen) .< valor
    labelArray = ImageMorphology.label_components(pixelesNegros) # etiqueta cada objeto de la imagen
    lengths = ImageMorphology.component_lengths(labelArray) # tamaños de los objetos
    mayorObjeto = maximum(lengths[2:end])
    etiquetasEliminar = findall(lengths .!= mayorObjeto) .- 1 # los objetos con menos de 1000 píxeles
    pixelesNegros2 = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray]; # se eliminan
    labelArray = ImageMorphology.label_components(pixelesNegros2); # recalcular objetos
    boundingBox = ImageMorphology.component_boxes(labelArray)[2]; # calcula la bounding box de cada objeto
    

    x1 = boundingBox[1][1]
    y1 = boundingBox[1][2]
    x2 = boundingBox[2][1]
    y2 = boundingBox[2][2]

    if(size(imagen[x1:x2,y1:y2]) < convert.(Int,round.(size(imagen).*0.73))) #tam box < tam imagen * 0.73
        return calculateAnchoMedio(imagen, valor+0.05)
    end

    imagen = Gray.(pixelesNegros2)[x1:x2,y1:y2]

    a = convert(Int,round(size(imagen, 1)/2))
    f = findfirst((x->x==1.0), imagen[a,:])
    l = findlast((x->x==1.0), imagen[a,:])
    #mosaicview(imagen; nrow=4, npad=10)

    return l-f

end;


# Cargar datos
numPeones = 50
numDamas = 53
numCaballos = 50
numAlfiles = 49
numTorres = 42
numReyes = 50
numPatterns = numPeones + numDamas + numCaballos + numAlfiles + numTorres+numReyes; # numero de fotos que tengamos disponibles
inputs = Array{Float32,2}(undef, numPatterns, 5);
targets = Array{String,1}(undef, numPatterns);
invalidInputs = Array{Int16,1}(undef, 0);

total = 0

damas = Array{Any,1}(undef, numDamas);
for i in 1:numDamas
    damas[i] = load("data/dama"*string(i)*".png")
    inputs[i,1] = calculateBoundingBox(damas[i])
    if (inputs[i,1] == 0) # en algunas fotos no calcula bien la bounding box
        push!(invalidInputs,i) # se guardan como inválidas para borrarlas
    end
    inputs[i,2] = calcularContorno(damas[i], 0.4)
    inputs[i,3] = calculateSimetriaX(damas[i], 0.4)
    inputs[i,4] = calculateSimetriaY(damas[i], 0.4)
    inputs[i,5] = calculateAnchoMedio(damas[i], 0.4)
    targets[total+i] = "dama"
end;

total += numDamas

peones = Array{Any,1}(undef, numPeones);
for i in 1:numPeones
    peones[i] = load("data/peon"*string(i)*".png")
    inputs[total+i,1] = calculateBoundingBox(peones[i])
    if (inputs[total+i,1] == 0)
        push!(invalidInputs,total+i)
    end
    inputs[total+i,2] = calcularContorno(peones[i], 0.4)
    inputs[total+i,3] = calculateSimetriaX(peones[i], 0.4)
    inputs[total+i,4] = calculateSimetriaY(peones[i], 0.4)
    inputs[total+i,5] = calculateAnchoMedio(peones[i], 0.4)
    targets[total+i] = "peón"
end;

total += numPeones

 caballos = Array{Any,1}(undef, numCaballos);
 for i in 1:numCaballos
     caballos[i] = load("data/caballo"*string(i)*".png")
     inputs[total+i,1] = calculateBoundingBox(caballos[i])
     if (inputs[total+i,1] == 0)
         push!(invalidInputs,total+i)
     end
     inputs[total+i,2] = calcularContorno(caballos[i], 0.4)
     inputs[total+i,3] = calculateSimetriaX(caballos[i], 0.4)
     inputs[total+i,4] = calculateSimetriaY(caballos[i], 0.4)
     inputs[total+i,5] = calculateAnchoMedio(caballos[i], 0.4)
     targets[total+i] = "caballo"
 end;

 total += numCaballos

 alfiles = Array{Any,1}(undef, numAlfiles);
 for i in 1:numAlfiles
     alfiles[i] = load("data/alfil"*string(i)*".png")
     inputs[total+i,1] = calculateBoundingBox(alfiles[i])
     if (inputs[total+i,1] == 0)
         push!(invalidInputs,total+i)
     end
     inputs[total+i,2] = calcularContorno(alfiles[i], 0.4)
     inputs[total+i,3] = calculateSimetriaX(alfiles[i], 0.4)
     inputs[total+i,4] = calculateSimetriaY(alfiles[i], 0.4)
     inputs[total+i,5] = calculateAnchoMedio(alfiles[i], 0.4)
     targets[total+i] = "alfil"
 end;

 total += numAlfiles

 torres = Array{Any,1}(undef, numTorres);
 for i in 1:numTorres
     torres[i] = load("data/torre"*string(i)*".png")
     inputs[total+i,1] = calculateBoundingBox(torres[i])
     if (inputs[total+i,1] == 0)
         push!(invalidInputs,total+i)
     end
     inputs[total+i,2] = calcularContorno(torres[i], 0.4)
     inputs[total+i,3] = calculateSimetriaX(torres[i], 0.4)
     inputs[total+i,4] = calculateSimetriaY(torres[i], 0.4)
     inputs[total+i,5] = calculateAnchoMedio(torres[i], 0.4)
     targets[total+i] = "torre"
 end;


 total += numTorres

 reyes = Array{Any,1}(undef, numReyes);
 for i in 1:numReyes
     reyes[i] = load("data/rey"*string(i)*".png")
     inputs[total+i,1] = calculateBoundingBox(reyes[i])
     if (inputs[total+i,1] == 0)
         push!(invalidInputs,total+i)
     end
     inputs[total+i,2] = calcularContorno(reyes[i], 0.4)
     inputs[total+i,3] = calculateSimetriaX(reyes[i], 0.4)
     inputs[total+i,4] = calculateSimetriaY(reyes[i], 0.4)
     inputs[total+i,5] = calculateAnchoMedio(reyes[i], 0.4)
     targets[total+i] = "rey"
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