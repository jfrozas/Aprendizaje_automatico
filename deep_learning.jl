using Random
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using FileIO
using Statistics: mean
using Images
include("funcionesF.jl")


# Cargar datos
numPeones = 50
numDamas = 50
numCaballos = 50
numAlfiles = 50
numTorres = 50
numReyes = 50
numPatterns = numPeones + numDamas + numCaballos + numAlfiles + numTorres + numReyes; # numero de fotos que tengamos disponibles
images = zeros(80, 80, numPatterns)
targets = zeros(numPatterns)
labels = 0:5; # dama = 0, peon = 1, caballo = 2, alfil = 3, torre = 4, rey = 5

total = 0

for i in 1:numDamas
    images[:,:,i+total] = convert.(Float32, Gray.(load("redimensionadas/dama"*string(i)*".png")))
    targets[i+total] = 0
end;

total += numDamas

for i in 1:numPeones
    images[:,:,i+total] = convert.(Float32, Gray.(load("redimensionadas/peon"*string(i)*".png")))
    targets[i+total] = 1
end;

total += numPeones

for i in 1:numCaballos
    images[:,:,i+total] = convert.(Float32, Gray.(load("redimensionadas/caballo"*string(i)*".png")))
    targets[i+total] = 2
end;

total += numCaballos

for i in 1:numAlfiles
    images[:,:,i+total] = convert.(Float32, Gray.(load("redimensionadas/alfil"*string(i)*".png")))
    targets[i+total] = 3
end;

total += numAlfiles

for i in 1:numTorres
    images[:,:,i+total] = convert.(Float32, Gray.(load("redimensionadas/torre"*string(i)*".png")))
    targets[i+total] = 4
end;

total += numReyes

for i in 1:numReyes
    images[:,:,i+total] = convert.(Float32, Gray.(load("redimensionadas/rey"*string(i)*".png")))
    targets[i+total] = 5
end;

function convertirArrayImagenesHWCN(imagenes)
    numPatrones = size(imagenes,3);
    nuevoArray = Array{Float32,4}(undef, 80, 80, 1, numPatrones);
    for i in 1:numPatrones
        @assert (size(imagenes[:,:,i])==(80,80)) "Las imagenes no tienen tamaño 80x80";
        nuevoArray[:,:,1,i] .= imagenes[:,:,i];
    end;
    return nuevoArray;
end;

images = convertirArrayImagenesHWCN(images);

seed!(40);

funcionTransferenciaCapasConvolucionales = relu;

# Definimos la red con la funcion Chain, que concatena distintas capas
function buildAnn(n::Int)
    Arquitecturas = Array{Chain,1}(undef, 8);
    Arquitecturas[1] = Chain(
        Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 80 -> 40
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 40 -> 20
        Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 20 -> 10
        x -> reshape(x, :, size(x, 4)),
        Dense(3200, 6), # 10x10x32
        softmax
    )
    Arquitecturas[2] = Chain(
        Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 80 -> 40
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 40 -> 20
        x -> reshape(x, :, size(x, 4)),
        Dense(12800, 6), # 20x20x32
        softmax
    )
    Arquitecturas[3] = Chain(
        Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 80 -> 40
        x -> reshape(x, :, size(x, 4)),
        Dense(25600, 6), # 40x40x16
        softmax
    )
    Arquitecturas[4] = Chain(
        Conv((3, 3), 1=>8, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 80 -> 40
        Conv((3, 3), 8=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 40 -> 20
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 20 -> 10
        x -> reshape(x, :, size(x, 4)),
        Dense(3200, 6), # 10x10x32
        softmax
    )
    Arquitecturas[5] = Chain(
        Conv((3, 3), 1=>8, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 80 -> 40
        Conv((3, 3), 8=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 40 -> 20
        x -> reshape(x, :, size(x, 4)),
        Dense(6400, 6), # 20x20x16
        softmax
    )
    Arquitecturas[6] = Chain(
        Conv((3, 3), 1=>8, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 80 -> 40
        x -> reshape(x, :, size(x, 4)),
        Dense(12800, 6), # 40x40x8
        softmax
    )
    Arquitecturas[7] = Chain(
        Conv((2, 2), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 80 -> 40
        Conv((2, 2), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 40 -> 20
        x -> reshape(x, :, size(x, 4)),
        Dense(12800, 6), # 20x20x32
        softmax
    )
    Arquitecturas[8] = Chain(
        Conv((2, 2), 1=>8, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 80 -> 40
        Conv((2, 2), 8=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)), # 40 -> 20
        x -> reshape(x, :, size(x, 4)),
        Dense(6400, 6), # 20x20x16
        softmax
    )
    return Arquitecturas[n]
end

# Definimos la funcion de loss de forma similar a las prácticas de la asignatura
loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

function calculateMetrics(out, targ)
    numClasses = size(targ,1)
    boolOutputs = classifyOutputs(out')'

    acc = zeros(numClasses)
    error = zeros(numClasses)
    sensitivity = zeros(numClasses)
    specificity = zeros(numClasses)
    vpp = zeros(numClasses)
    vpn = zeros(numClasses)
    f1 = zeros(numClasses)

    for class in 1:numClasses
        v = confusionMatrix(boolOutputs[class,:],targ[class,:])
        acc[class] = v[1]
        error[class] = v[2]
        sensitivity[class] = v[3]
        specificity[class] = v[4]
        vpp[class] = v[5]
        vpn[class] = v[6]
        f1[class] = v[7]
    end

    return (mean(acc), mean(error), mean(sensitivity), mean(specificity),
            mean(vpp), mean(vpn), mean(f1))
end


println("Comenzando entrenamiento...")
numExecutions = 1;
maxEpochs = 1000;
learningRate = 0.001;
numFolds = 10;

crossValidationIndices = crossvalidation(targets, numFolds);

function modelCrossValidation(inputs::Array{Float32,4}, targ::AbstractArray{Bool,2}, crossValidationIndices::Array{Int64,1}, arquitectura::Int)

    @assert(size(inputs,4) == size(targ,2));

    acc = Array{Float64,1}(undef, numFolds);
    err = Array{Float64,1}(undef, numFolds);
    sen = Array{Float64,1}(undef, numFolds);
    spe = Array{Float64,1}(undef, numFolds);
    ppv = Array{Float64,1}(undef, numFolds);
    npv = Array{Float64,1}(undef, numFolds);
    f1  = Array{Float64,1}(undef, numFolds);

    for numFold in 1:numFolds

        trainingInputs  = inputs[:,:,:,crossValidationIndices.!=numFold];
        testInputs      = inputs[:,:,:,crossValidationIndices.==numFold];
        trainingTargets = targ[:,crossValidationIndices.!=numFold];
        testTargets     = targ[:,crossValidationIndices.==numFold];

        train_set = [(trainingInputs, trainingTargets)]
    
        accEachRep = Array{Float64,1}(undef, numExecutions);
        errEachRep = Array{Float64,1}(undef, numExecutions);
        senEachRep = Array{Float64,1}(undef, numExecutions);
        speEachRep = Array{Float64,1}(undef, numExecutions);
        ppvEachRep = Array{Float64,1}(undef, numExecutions);
        npvEachRep = Array{Float64,1}(undef, numExecutions);
        f1EachRep  = Array{Float64,1}(undef, numExecutions);

        for numTraining in 1:numExecutions

            global numCicloUltimaMejora, numCiclo, mejorPrecision, criterioFin;
            numCiclo = 0
            numCicloUltimaMejora = 0
            criterioFin = false
            mejorPrecision = -Inf # de entrenamiento
            accTest = 0; errTest = 0; senTest = 0; speTest = 0; ppvTest = 0; npvTest = 0; f1Test = 0

            # Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
            opt = ADAM(learningRate);

            global ann = buildAnn(arquitectura)

            while (!criterioFin)

                # Se entrena un ciclo
                Flux.train!(loss, Flux.params(ann), train_set, opt);
            
                numCiclo += 1;
            
                # Se calcula la precision en el conjunto de entrenamiento:
                precisionEntrenamiento, = calculateMetrics(ann(trainingInputs), trainingTargets)
            
                # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test
                if (precisionEntrenamiento > mejorPrecision)
                    mejorPrecision = precisionEntrenamiento;
                    accTest, errTest, senTest, speTest, ppvTest, npvTest, f1Test = calculateMetrics(ann(testInputs), testTargets)
                    numCicloUltimaMejora = numCiclo;
                end
            
                # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
                if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
                    opt.eta /= 10.0
                    numCicloUltimaMejora = numCiclo;
                end
            
                # Criterios de parada:
                if (precisionEntrenamiento >= 0.995 || (precisionEntrenamiento >= 0.9 && accTest == 1) || numCiclo - numCicloUltimaMejora >= 10)
                    criterioFin = true;
                end
            end
            accEachRep[numTraining] = accTest
            errEachRep[numTraining] = errTest
            senEachRep[numTraining] = senTest
            speEachRep[numTraining] = speTest
            ppvEachRep[numTraining] = ppvTest
            npvEachRep[numTraining] = npvTest
            f1EachRep[numTraining] = f1Test

        end;

        acc[numFold] = mean(accEachRep)
        err[numFold] = mean(errEachRep)
        sen[numFold] = mean(senEachRep)
        spe[numFold] = mean(speEachRep)
        ppv[numFold] = mean(ppvEachRep)
        npv[numFold] = mean(npvEachRep)
        f1[numFold] = mean(f1EachRep)
        println("\nFold ", numFold, ":")
        println("Precision = ", acc[numFold])

    end; 

    return (mean(acc), mean(err), mean(sen), mean(spe), mean(ppv), mean(npv), mean(f1));

end;

metrics = zeros(7,8)

targets = oneHotEncoding(targets)
for i in 1:8
    acc, err, sen, spe, ppv, npv, f1 = modelCrossValidation(copy(images), copy(targets'), crossValidationIndices, i)
    metrics[1,i] = acc
    metrics[2,i] = err
    metrics[3,i] = sen
    metrics[4,i] = spe
    metrics[5,i] = ppv
    metrics[6,i] = npv
    metrics[7,i] = f1
    println("\nArquitectura ", i, ":")
    println("Precision: ", acc)
    println("Tasa de fallo: ", err)
    println("Sensibilidad: ", sen)
    println("Especificidad: ", spe)
    println("Valor predictivo positivo: ", ppv)
    println("Valor predictivo negativo: ", npv)
    println("F1-score: ", f1, "\n\n")
end

# tabla de latex con los resultados
println("\n\\begin{table}[H]")
println("\\renewcommand{\\arraystretch}{1.5}")
println("\\centering")
println("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|} \\hline")
println("\\multicolumn{9}{|c|}{\\textbf{Resultados de Deep Learning}} \\\\ \\hline")
println("Arquitectura & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\\\ \\hline")
print("Precisión     & "); @printf("%.4f", metrics[1,1]); print(" & "); @printf("%.4f", metrics[1,2]); print(" & "); @printf("%.4f", metrics[1,3]); print(" & "); @printf("%.4f", metrics[1,4]);
    print(" & "); @printf("%.4f", metrics[1,5]); print(" & "); @printf("%.4f", metrics[1,6]); print(" & "); @printf("%.4f", metrics[1,7]); print(" & "); @printf("%.4f", metrics[1,8]); print(" \\\\\n")
print("Tasa de fallo & "); @printf("%.4f", metrics[2,1]); print(" & "); @printf("%.4f", metrics[2,2]); print(" & "); @printf("%.4f", metrics[2,3]); print(" & "); @printf("%.4f", metrics[2,4]);
    print(" & "); @printf("%.4f", metrics[2,5]); print(" & "); @printf("%.4f", metrics[2,6]); print(" & "); @printf("%.4f", metrics[2,7]); print(" & "); @printf("%.4f", metrics[2,8]); print(" \\\\\n")
print("Sensibilidad  & "); @printf("%.4f", metrics[3,1]); print(" & "); @printf("%.4f", metrics[3,2]); print(" & "); @printf("%.4f", metrics[3,3]); print(" & "); @printf("%.4f", metrics[3,4]);
    print(" & "); @printf("%.4f", metrics[3,5]); print(" & "); @printf("%.4f", metrics[3,6]); print(" & "); @printf("%.4f", metrics[3,7]); print(" & "); @printf("%.4f", metrics[3,8]); print(" \\\\\n")
print("Especificidad & "); @printf("%.4f", metrics[4,1]); print(" & "); @printf("%.4f", metrics[4,2]); print(" & "); @printf("%.4f", metrics[4,3]); print(" & "); @printf("%.4f", metrics[4,4]);
    print(" & "); @printf("%.4f", metrics[4,5]); print(" & "); @printf("%.4f", metrics[4,6]); print(" & "); @printf("%.4f", metrics[4,7]); print(" & "); @printf("%.4f", metrics[4,8]); print(" \\\\\n")
print("PV+           & "); @printf("%.4f", metrics[5,1]); print(" & "); @printf("%.4f", metrics[5,2]); print(" & "); @printf("%.4f", metrics[5,3]); print(" & "); @printf("%.4f", metrics[5,4]);
    print(" & "); @printf("%.4f", metrics[5,5]); print(" & "); @printf("%.4f", metrics[5,6]); print(" & "); @printf("%.4f", metrics[5,7]); print(" & "); @printf("%.4f", metrics[5,8]); print(" \\\\\n")
print("PV-           & "); @printf("%.4f", metrics[6,1]); print(" & "); @printf("%.4f", metrics[6,2]); print(" & "); @printf("%.4f", metrics[6,3]); print(" & "); @printf("%.4f", metrics[6,4]);
    print(" & "); @printf("%.4f", metrics[6,5]); print(" & "); @printf("%.4f", metrics[6,6]); print(" & "); @printf("%.4f", metrics[6,7]); print(" & "); @printf("%.4f", metrics[6,8]); print(" \\\\\n")
print("F1-score      & "); @printf("%.4f", metrics[7,1]); print(" & "); @printf("%.4f", metrics[7,2]); print(" & "); @printf("%.4f", metrics[7,3]); print(" & "); @printf("%.4f", metrics[7,4]);
    print(" & "); @printf("%.4f", metrics[7,5]); print(" & "); @printf("%.4f", metrics[7,6]); print(" & "); @printf("%.4f", metrics[7,7]); print(" & "); @printf("%.4f", metrics[7,8]); print(" \\\\ \\hline\n")
println("\\end{tabular}")
println("\\caption{Parámetros y métricas de evaluación de Deep Learning.}")
println("\\end{table}")
