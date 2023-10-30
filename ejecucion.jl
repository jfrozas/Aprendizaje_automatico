include("aprox2.jl")

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

c = Array{Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64}}(undef,8)
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
    c[i] = round.(j[1:7], digits=4)
    

end

println("\\begin{table}[ht]")
println("\\renewcommand{\\arraystretch}{1.5}")
println("\\centering")
println("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
println("\\hline")
println("\\multicolumn{9}{|c|}{\\textbf{Resultados de Redes Neuronales}} \\\\ \\hline")
println("& \\multicolumn{8}{c|}{\\textbf{Configuraciones probadas}} \\\\ \\hline")
println("Topología & $(topology[1]) & $(topology[2]) & $(topology[3]) & $(topology[4]) & $(topology[5]) & $(topology[6]) & $(topology[7]) & $(topology[8])\\\\ \\hline")
println("& \\multicolumn{8}{c|}{\\textbf{Valores obtenidos}} \\\\ \\hline")
println("Precisión                 & $(c[1][1]) & $(c[2][1]) & $(c[3][1]) & $(c[4][1]) & $(c[5][1]) & $(c[6][1]) & $(c[7][1]) & $(c[7][1]) \\\\")
println("Tasa de fallo             & $(c[1][2]) & $(c[2][2]) & $(c[3][2]) & $(c[4][2]) & $(c[5][2]) & $(c[6][2]) & $(c[7][2]) & $(c[7][2]) \\\\")
println("Sensibilidad              & $(c[1][3]) & $(c[2][3]) & $(c[3][3]) & $(c[4][3]) & $(c[5][3]) & $(c[6][3]) & $(c[7][3]) & $(c[7][3]) \\\\")
println("Especificidad             & $(c[1][4]) & $(c[2][4]) & $(c[3][4]) & $(c[4][4]) & $(c[5][4]) & $(c[6][4]) & $(c[7][4]) & $(c[7][4]) \\\\")
println("PV+                       & $(c[1][5]) & $(c[2][5]) & $(c[3][5]) & $(c[4][5]) & $(c[5][5]) & $(c[6][5]) & $(c[7][5]) & $(c[7][5]) \\\\")
println("PV-                       & $(c[1][6]) & $(c[2][6]) & $(c[3][6]) & $(c[4][6]) & $(c[5][6]) & $(c[6][6]) & $(c[7][6]) & $(c[7][6]) \\\\")
println("F1-score                  & $(c[1][7]) & $(c[2][7]) & $(c[3][7]) & $(c[4][7]) & $(c[5][7]) & $(c[6][7]) & $(c[7][7]) & $(c[7][7]) \\\\ \\hline")
println("\\end{tabular}")
println("\\caption{Parámetros y métricas de evaluación de las redes neuronales.}")
println("\\end{table}")

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
    c[i] = round.(j[1:7], digits=4)

    
end

println("\\begin{table}[!]")
println("\\renewcommand{\\arraystretch}{1.5}")
println("\\centering")
println("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
println("\\hline")
println("\\multicolumn{9}{|c|}{\\textbf{Resultados de SVM}} \\\\ \\hline")
println("& \\multicolumn{8}{c|}{\\textbf{Configuraciones probadas}} \\\\ \\hline")
println("Kernel & $(kernel[1]) & $(kernel[2]) & $(kernel[3]) & $(kernel[4]) & $(kernel[5]) & $(kernel[6]) & $(kernel[7]) & $(kernel[8]) \\\\ \\hline")
println("C & $(C[1]) & $(C[2]) & $(C[3]) & $(C[4]) & $(C[5]) & $(C[6]) & $(C[7]) & $(C[8]) \\\\ \\hline")
println("& \\multicolumn{8}{c|}{\\textbf{Valores obtenidos}} \\\\ \\hline")
println("Precisión & $(c[1][1]) & $(c[2][1]) & $(c[3][1]) & $(c[4][1]) & $(c[5][1]) & $(c[6][1]) & $(c[7][1]) & $(c[7][1]) \\\\")
println("Tasa de fallo & $(c[1][2]) & $(c[2][2]) & $(c[3][2]) & $(c[4][2]) & $(c[5][2]) & $(c[6][2]) & $(c[7][2]) & $(c[7][2]) \\\\")
println("Sensibilidad & $(c[1][3]) & $(c[2][3]) & $(c[3][3]) & $(c[4][3]) & $(c[5][3]) & $(c[6][3]) & $(c[7][3]) & $(c[7][3]) \\\\")
println("Especificidad & $(c[1][4]) & $(c[2][4]) & $(c[3][4]) & $(c[4][4]) & $(c[5][4]) & $(c[6][4]) & $(c[7][4]) & $(c[7][4]) \\\\")
println("PV+ & $(c[1][5]) & $(c[2][5]) & $(c[3][5]) & $(c[4][5]) & $(c[5][5]) & $(c[6][5]) & $(c[7][5]) & $(c[7][5]) \\\\")
println("PV- & $(c[1][6]) & $(c[2][6]) & $(c[3][6]) & $(c[4][6]) & $(c[5][6]) & $(c[6][6]) & $(c[7][6]) & $(c[7][6]) \\\\")
println("F1-score & $(c[1][7]) & $(c[2][7]) & $(c[3][7]) & $(c[4][7]) & $(c[5][7]) & $(c[6][7]) & $(c[7][7]) & $(c[7][7]) \\\\ \\hline")
println("\\end{tabular}")
println("\\caption{Parámetros y métricas de evaluación de SVM.}")
println("\\end{table}")


# Entrenamos los arboles de decision
for i in 1:6
    println("\n\n\nDecisionTree")
    print("MaxDepth: ")
    println(maxDepth[i])
    println("")
    j = modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth[i]), finalInputs, finalTargets, crossValidationIndices);
    printMetrics(j, numClasses)
    c[i] = round.(j[1:7], digits=4)

end

    println("\\begin{table}[!]")
    println("\\renewcommand{\\arraystretch}{1.5}")
    println("\\centering")
    println("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    println("\\hline")
    println("\\multicolumn{7}{|c|}{\\textbf{Resultados de Árboles de Decisión}} \\\\ \\hline")
    println("& \\multicolumn{6}{c|}{\\textbf{Configuraciones probadas}} \\\\ \\hline")
    println("MaxDepth & $(maxDepth[1]) & $(maxDepth[2]) & $(maxDepth[3]) & $(maxDepth[4]) & $(maxDepth[5]) & $(maxDepth[6]) \\\\ \\hline")
    println("& \\multicolumn{6}{c|}{\\textbf{Valores obtenidos}} \\\\ \\hline")
    println("Precisión & $(c[1][1]) & $(c[2][1]) & $(c[3][1]) & $(c[4][1]) & $(c[5][1]) & $(c[6][1]) \\\\")
    println("Tasa de fallo & $(c[1][2]) & $(c[2][2]) & $(c[3][2]) & $(c[4][2]) & $(c[5][2]) & $(c[6][2]) \\\\")
    println("Sensibilidad & $(c[1][3]) & $(c[2][3]) & $(c[3][3]) & $(c[4][3]) & $(c[5][3]) & $(c[6][3]) \\\\")
    println("Especificidad & $(c[1][4]) & $(c[2][4]) & $(c[3][4]) & $(c[4][4]) & $(c[5][4]) & $(c[6][4]) \\\\")
    println("PV+ & $(c[1][5]) & $(c[2][5]) & $(c[3][5]) & $(c[4][5]) & $(c[5][5]) & $(c[6][5]) \\\\")
    println("PV- & $(c[1][6]) & $(c[2][6]) & $(c[3][6]) & $(c[4][6]) & $(c[5][6]) & $(c[6][6]) \\\\")
    println("F1-score & $(c[1][7]) & $(c[2][7]) & $(c[3][7]) & $(c[4][7]) & $(c[5][7]) & $(c[6][7]) \\\\ \\hline")
    println("\\end{tabular}")
    println("\\caption{Parámetros y métricas de evaluación de los árboles de decisión.}")
    println("\\end{table}")


# Entrenamos los kNN
for i in 1:6
    println("\n\n\nkNN")
    print("NumNeighbors: ")
    println(numNeighbors[i])
    println("")
    j = modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors[i]), finalInputs, finalTargets, crossValidationIndices);
    printMetrics(j, numClasses)
    c[i] = round.(j[1:7], digits=4)

    

end


println("\\begin{table}[!]")
    println("\\renewcommand{\\arraystretch}{1.5}")
    println("\\centering")
    println("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    println("\\hline")
    println("\\multicolumn{7}{|c|}{\\textbf{Resultados de kNN}} \\\\ \\hline")
    println("& \\multicolumn{6}{c|}{\\textbf{Configuraciones probadas}} \\\\ \\hline")
    println("MaxDepth & $(numNeighbors[1]) & $(numNeighbors[2]) & $(numNeighbors[3]) & $(numNeighbors[4]) & $(numNeighbors[5]) & $(numNeighbors[6]) \\\\ \\hline")
    println("& \\multicolumn{6}{c|}{\\textbf{Valores obtenidos}} \\\\ \\hline")
    println("Precisión & $(c[1][1]) & $(c[2][1]) & $(c[3][1]) & $(c[4][1]) & $(c[5][1]) & $(c[6][1]) \\\\")
    println("Tasa de fallo & $(c[1][2]) & $(c[2][2]) & $(c[3][2]) & $(c[4][2]) & $(c[5][2]) & $(c[6][2]) \\\\")
    println("Sensibilidad & $(c[1][3]) & $(c[2][3]) & $(c[3][3]) & $(c[4][3]) & $(c[5][3]) & $(c[6][3]) \\\\")
    println("Especificidad & $(c[1][4]) & $(c[2][4]) & $(c[3][4]) & $(c[4][4]) & $(c[5][4]) & $(c[6][4]) \\\\")
    println("PV+ & $(c[1][5]) & $(c[2][5]) & $(c[3][5]) & $(c[4][5]) & $(c[5][5]) & $(c[6][5]) \\\\")
    println("PV- & $(c[1][6]) & $(c[2][6]) & $(c[3][6]) & $(c[4][6]) & $(c[5][6]) & $(c[6][6]) \\\\")
    println("F1-score & $(c[1][7]) & $(c[2][7]) & $(c[3][7]) & $(c[4][7]) & $(c[5][7]) & $(c[6][7]) \\\\ \\hline")
    println("\\end{tabular}")
    println("\\caption{Parámetros y métricas de evaluación de los árboles de decisión.}")
    println("\\end{table}")