using CSV
using DataFrames
using ArgParse
using Distributed
using SymbolicRegression

using PyCall
#sympy = pyimport("sympy")

# example call of this wrapper:
# julia adsorptionSR.jl --help
# julia adsorptionSR.jl adsorptionDatasets/Langmuir1918methane.csv --thermo=true

ENV["JULIA_WORKER_TIMEOUT"] = 360.0

# get commandline args
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "file"
            help = "dataset to run SymbolicRegression on (should be .csv)"
            required = true

        "--t1"
            help = "the first thermo constraint penalty"
            arg_type = Float64
            default = 1.0
        "--t2"
            help = "the second thermo constraint penalty"
            arg_type = Float64
            default = 1.0
        "--t3"
            help = "the third thermo constraint penalty"
            arg_type = Float64
            default = 1.0
        "--npopulations"
            help = "number of populations to simulate"
            arg_type = Int
            default = 16
        "--numprocs"
            help = "number of processes to work on SR in parallel"
            arg_type = Int
            default = 8
        "--crossover"
            help = "% chance of new members of population being created by randomly merging two previous membrs"
            arg_type = Float32
            default = 0.0f0
        "--startVars"
            help = "the starting index/column of the variables to be included from the dataset.  
                If using thermo constraints, the first var is expected to be pressure"
            arg_type = Int
            default = 1
        "--stopVars"
            help = "the ending index/column (inclusive) of the variables to be included from the dataset"
            arg_type = Int
            default = 1
        "--predVar"
            help = "the index/column of the variable to be predicted in the dataset. 
                This does need to be within the range of variables to be included for SR but will be removed from that range if it is"
            arg_type = Int
            default = 2

        "--niterations"
            help = "number of iterations of genetic algorithm to run"
            arg_type = Int
            default = 10
        "--numRuns"
            help = "number of runs with the above parameters (will be added sequentially to the same output file)"
            arg_type = Int
            default = 1
    end
            
    return parse_args(s)
end

function main()

    # get commandline args
    args = parse_commandline()

    # import data and turn into dataframe
    file = CSV.File(args["file"])
    data = DataFrame(file)
    
    #if args["thermo"]
    #    varMap = ["p"]
    #else
    #    varMap = names(data)
    #    println(varMap)
    #end
    varMap = names(data)
    println(varMap)

    # range of vars to collect
    # if one var it would be (n - n) so need +1 (because bounds are inclusive)
    numVars = abs(args["stopVars"] - args["startVars"]) + 1
    
    predInRange = false
    # if pred var in range of vars to use
    if (args["predVar"] >= args["startVars"] && args["predVar"] <= args["stopVars"])
        # keep it out of count of total vars to collect
        numVars -= 1
        predInRange = true
        println("pred var is in range of vars to use")

        # if the only var in the range is the var to be predicited 
        if numVars == 0
            println("pred var is only var in range!")
            # quit program since SR will have no X to work with
            exit()
        end
    end
    # if just one var to use in SR
    if numVars == 1
        # grab each col, transpose and cast to array
        X = Matrix{Float64}(data[:,args["startVars"]]')
        y = data[:,args["predVar"]]
    
    # otherwise there are multiple cols to grab from dataframe
    else
        y = data[:,args["predVar"]] 
        # if y is in range of vars to be used by SR, remove it before selecting them
        if predInRange
            data = data[:, Not(args["predVar"])]
            # decrement to account for range change
            args["stopVars"] -= 1
        end
        # select range of vars
        X = permutedims(Matrix(data[:, Between(args["startVars"], args["stopVars"])]))

    end

    println(typeof(X))
    println(typeof(y))
    println(size(X))
    println(size(y))
    println(X)
    println(y)

    if (args["t1"] != 1 || args["t2"] != 1 || args["t3"] != 1)
        penalties = [args["t1"], args["t2"], args["t3"]]
        println(penalties)
    else
        penalties = nothing
    end

    options = SymbolicRegression.Options(
        #verbosity=1,
        binary_operators=(+, *, /, -),
        unary_operators=(),
        npopulations=args["npopulations"],
        progress=false,
        crossoverProbability=args["crossover"],
        penalties = penalties
        #recorder=false # crashed my computer when enabled...
    )

    for i = 1:args["numRuns"]
        # total iterations = niterations * npopulations
        # diversity and range of complexity preportional to npopulations
        println("run ", i)
        @time hallOfFame = EquationSearch(X, y, varMap=varMap, niterations=args["niterations"], 
                        options=options, numprocs=args["numprocs"])# multithreading=true)
        
        #println("HOF: \n")
        #println(hallOfFame)
        dominating = calculate_pareto_frontier(X, y, hallOfFame, options)
    end
                        #, numprocs=args["numprocs"])# 
end

main()