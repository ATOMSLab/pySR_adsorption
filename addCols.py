import re
import sys
import pandas as pd
import sympy

assuming_positive_pressure = True

def simplify_expr(expr, variables=[], parameters=[]):
    # expr is sympy operation
    # parameters: list of parameters in sympy format
    if assuming_positive_pressure and variables: # for SymPy to simplify sqrt(p)**2 
        expr = expr.subs(variables[0], sympy.symbols(str(variables[0]),positive = True))
    prime_num = [5., 7., 11., 13., 17., 19., 23., 29., 31., 37., 41., 43., 47., 53., 59., 61., 67., 71., 73., 79., 83.]
    prime_list = prime_num[0:len(parameters)]
    # substitute and simplify
    sub_expr = expr.subs([(parameters[i], prime_list[i]) for i in range(len(prime_list))]).evalf()
    #print("original expr =", expr)
    #print("sub_expr =", sub_expr)
    try:
        # this is only for rational function, simplify the coefficient of the highest degree term
        if sub_expr.is_rational_function() \
            and ('DIV' in [i.name for i in sympy.count_ops(sub_expr,visual=True).free_symbols]): # SymPy classifies polynomial as rational
            # splitting numerator and denominator
            num, den = sub_expr.as_numer_denom()
            #print("(num) / (den) = (", num, ") / (", den, ")")
            # finding leading coefficient of num and den
            if not num.is_constant():
                #print(type(variables[0]))
                num_degree = sympy.degree(num, gen=variables[0])
                #print("here")
                for i in sympy.LT(num).atoms(sympy.Number):
                    lead_num = i
            else:
                num_degree = 0
                lead_num = num
            
            den_degree = sympy.degree(den, gen=variables[0])
            #print(den)
            den = sympy.Poly(den)
            lead_den = den.all_coeffs()[0]
            # compare the degree to decide the factor for simplification
            if num_degree > den_degree:
                factor = lead_num
            else:
                factor = lead_den
            sub_expr = (sympy.expand(num/factor))/(sympy.expand(den/factor))
            sub_expr = sub_expr.subs(1.0, 1) # SymPy doesn't reduce 1 if it is float --> convert to integer
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
        print("with constants")
    # obtain the numerical parameter after simplifying the expression
    # num_val = [a for a in sub_expr.atoms() if not a.is_Symbol]
    #print(sub_expr)
    num_val = [a for a in sub_expr.atoms(sympy.Number)]
    # remove -1 if there is a division in the expression
    # remove 2 or -2 if pow2 presents, remove 3 or -3 if pow3 presents
    for element in [1, -1, 2, -2, 3, -3, sympy.Rational(1, 2), sympy.Rational(-1, 2)]:
        if element in num_val:
            num_val.remove(element)
    # substitute back the parameter
    constant_list = ['_c' + str(i) + '_' for i in range(len(num_val))]
    cansp = sub_expr.subs([(num_val[i], constant_list[i]) for i in range(len(num_val))])
    # rearrange constants in order of subcription number (adapted from BMS original code)
    can = str(cansp)
    ps = list([str(s) for s in cansp.free_symbols])
    positions = []
    for p in ps:
        if p.startswith('_') and p.endswith('_'):
            positions.append((can.find(p), p))
    positions.sort()
    pcount = 1
    for pos, p in positions:
        can = can.replace(p, 'c%d' % pcount)
        pcount += 1
    return can
    #return sub_expr

def simplify_expr_num(expr, variables=[], parameters=[]):
    if type(expr) == str:
        expr = sympy.parse_expr(expr)
    #variables = [symbols(v) for v in variables]
    # expr is sympy operation
    # parameters: list of parameters in sympy format
    if assuming_positive_pressure and variables: # for SymPy to simplify sqrt(p)**2 
        expr = expr.subs(variables[0],sympy.symbols(str(variables[0]),positive = True))
    prime_num = [5., 7., 11., 13., 17., 19., 23., 29., 31., 37., 41., 43., 47., 53., 59., 61., 67., 71., 73., 79., 83.]
    prime_list = prime_num[0:len(parameters)]
    # substitute and simplify
    sub_expr = expr.subs([(parameters[i], prime_list[i]) for i in range(len(prime_list))]).evalf()
    #print("original expr =", expr)
    #print("sub_expr =", sub_expr)
    #print("here now")
    try:
        # this is only for rational function, simplify the coefficient of the highest degree term
        if sub_expr.is_rational_function() \
            and ('DIV' in [i.name for i in sympy.count_ops(sub_expr,visual=True).free_symbols]): # SymPy classifies polynomial as rational
            # splitting numerator and denominator
            num, den = sub_expr.as_numer_denom()
            #print("(num) / (den) = (", num, ") / (", den, ")")
            # finding leading coefficient of num and den
            if not num.is_constant():
                #print(type(variables[0]))
                num_degree = sympy.degree(num, gen=variables[0])
                for i in sympy.LT(num).atoms(sympy.Number):
                    lead_num = i
            else:
                num_degree = 0
                lead_num = num
            den_degree = sympy.degree(den, gen=variables[0])
            den = sympy.Poly(den)
            lead_den = den.all_coeffs()[0]
            # compare the degree to decide the factor for simplification
            if num_degree > den_degree:
                factor = lead_num
            else:
                factor = lead_den
            sub_expr = (sympy.expand(num/factor))/(sympy.expand(den/factor))
            sub_expr = sub_expr.subs(1.0, 1) # SymPy doesn't reduce 1 if it is float --> convert to integer
    except:
        print('Error in simplifying this rational function to reduce 1 parameter')
    # obtain the numerical parameter after simplifying the expression
    # num_val = [a for a in sub_expr.atoms() if not a.is_Symbol]
    #print(sub_expr)
    num_val = [a for a in sub_expr.atoms(sympy.Number)]
    # remove -1 if there is a division in the expression
    # remove 2 or -2 if pow2 presents, remove 3 or -3 if pow3 presents
    for element in [1, -1, 2, -2, 3, -3, sympy.Rational(1, 2), sympy.Rational(-1, 2)]:
        if element in num_val:
            num_val.remove(element)
    # substitute back the parameter
    #constant_list = ['_c' + str(i) + '_' for i in range(len(num_val))]
    #cansp = sub_expr.subs([(num_val[i], constant_list[i]) for i in range(len(num_val))])
    # rearrange constants in order of subcription number (adapted from BMS original code)
    #can = str(cansp)
    #ps = list([str(s) for s in cansp.free_symbols])
    #positions = []
    #for p in ps:
    #    if p.startswith('_') and p.endswith('_'):
    #        positions.append((can.find(p), p))
    #positions.sort()
    #pcount = 1
    #for pos, p in positions:
    #    can = can.replace(p, 'c%d' % pcount)
    #    pcount += 1
    #return can
    return sub_expr

def replaceMathFunctions(expr):
    expr = str(expr)
    functions = [("cube", "**3"), ("square", "**2")]
    
    if not any([f[0] in expr for f in functions]):
        return expr

    else:
        tries = 5
        for f in functions:
            while (f[0] in expr) and (tries > 0):
                start = expr.find(f[0])
                par = 0
                if expr[start+len(f[0])] != "(":
                    print("failed to parse")
                    return
                else:
                    par += 1
                i = start + len(f[0]) + 1
                #print("starting i =", i)
                # while not to end or not found end of parens
                while (i < len(expr)) and (par > 0):
                    char = expr[i]
                    if char == ")":
                        par -= 1
                    elif char == "(":
                        par += 1

                    if par == 0:
                        expr = list(expr)
                        expr.insert(i, f[1])
                        #print(expr)
                        expr = "".join(expr)
                        expr = expr.replace(f[0], "", 1)
                        #print(expr)

                    i += 1
                tries -= 1

        return expr

# turn string with floats into same string but each is replaced with c1, c2, ...
def floatsToConstants(expr):
    if type(expr) == str:
        expr = sympy.parse_expr(expr)

    # find all floating point numbers
    # from python docs 
    finder = re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')
    newExpr = str(expr)
    newExpr = re.sub(finder, "c", newExpr)
    parameters = []

    for i in range(len(re.findall(finder, str(expr)))):
        newExpr = re.sub(r'c(?!\d)', "c{}".format(i), newExpr, 1)
        parameters.append("c{}".format(i))

    return newExpr, parameters

# count occurences of operators in string to calculate complexity
def findComplexity(expr):
    expr = str(expr)
    expr = expr.replace("**", "^")
    expr = expr.replace("sqrt_abs", "sqrt")
    binaryFinder = re.compile(r'[\+\-\*\/\^]') # WILL NOT WORK FOR UNARY
    binaryOps = len(re.findall(binaryFinder, expr))
    #print("num binary ops =", binaryOps)

    unaryFinder = re.compile(r'square')
    unaryOps = len(re.findall(unaryFinder, expr))
    #print("squares =", unaryOps)
    unaryFinder = re.compile(r'cube')
    unaryOps += len(re.findall(unaryFinder, expr))
    #print("cubes =", unaryOps)
    unaryFinder = re.compile(r'sqrt')
    unaryOps += len(re.findall(unaryFinder, expr))
    #print("sqrt_abs =", unaryOps)
    #print("num unary ops =", unaryOps)

    return 2 * binaryOps + 1 + unaryOps

def getCanonical(expr):
    # replace sqrt_abs with sqrt
    expr = str(expr)
    expr = expr.replace("sqrt_abs", "sqrt")
    expr = replaceMathFunctions(expr)

    # turn string with numbers into one with constant labels
    # maybe try keep numbers here?
    ex, parameters = floatsToConstants(expr)
    #print(ex)
    simplified = ""

    try:
        ex = sympy.sympify(ex)
        atomd = dict([(a.name, a) for a in ex.atoms() if a.is_Symbol])
        variables = [sympy.symbols("p")]
        parameters = parameters
        #print(parameters)
        can = simplify_expr(ex, variables, parameters) # simplify the expression
        #print("expr =", expr, "and vars =", variables)
        simplified = simplify_expr_num(expr, variables)
        #print("also here")
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
        print("failed to get canonical fully")
        #print(ex)
        can = str(ex)

    complexity = findComplexity(can)
    
    return [str(can), parameters, complexity, str(simplified)]

# adds canonical form, simplified form and complexity to df
def addCanonical(run):
    #newData = [getCanonical(expr) for expr in run["equation"]]
    #newData = []
    #for expr in run["equation"]:
    #    newData.append(getCanonical(expr))
    newData = map(getCanonical, run["equation"])
    #print(len(newData))
    run[["canonEquation", "parameters", "canonComplexity", "simplified"]] = pd.DataFrame(newData, index=run.index)
    return run

    # remove expressions with identical canonical form (keep one with best loss)
def removeRepeats(run):
    #print(run.applymap(lambda x: isinstance(x, list)).all())
    #run["parameters"] = run["parameters"].astype("str")

    # make sure all loss is numbers because apparently need to worry about that...  
    run["loss"] = pd.to_numeric(run["loss"])

    run.drop(run[run["loss"] == 0].index, inplace=True)

    print(len(run), "-> ", end="")
    unique = pd.DataFrame([])
    for index, row in run.iterrows():

        # get canonical form for this row
        expr = row["canonEquation"]
        # find best performing row with this canonical form
        allInstances = run.loc[run["canonEquation"] == expr]
        bestExpr = allInstances.loc[allInstances["loss"].idxmin()]
        #print("adding", bestExpr["equation"])
        # add it to list
        unique = unique.append(bestExpr)

    unique.drop_duplicates(subset=["canonEquation"], inplace=True)
    #print(unique.head())
    #print(unique["canonEquation"])
    print(len(unique))
    return unique

import sympy
def is_monotonic_increasing(expr, interval, var):

    # constant value never decreases    
    if expr.is_constant():
        return True

    # get critical points as list
    turning_points = list(sympy.solveset(expr.diff(var), var, interval))
    turning_points.sort()
    # failed to find critical points
    # there could be 0 or infinite...
    if (turning_points == []):
        # fall back to simpler increasing function
        return bool(1 if (expr.limit(var, interval.end) - expr.limit(var, interval.start)) >= 0 else 0)
    increasing = 1
    # turn to false if interval from start of main interval to first critical point not increasing
    increasing = min(increasing, (1 if (expr.limit(var, turning_points[0]) - expr.limit(var, interval.start)) >= 0 else 0))
    # check intervals between all critical points
    for i in range(len(turning_points)-1):
        thisPoint = turning_points[i]
        nextPoint = turning_points[i+1]
        increasing = min(increasing, (1 if (expr.limit(var, nextPoint) - expr.limit(var, thisPoint)) >= 0 else 0))
        #increasing = min(increasing, sympy.is_increasing(expr, sympy.Interval(thisPoint, nextPoint, false, false), var))
    # check last interval
    increasing = min(increasing, (1 if (expr.limit(var, interval.end) - expr.limit(var, turning_points[-1])) >= 0 else 0))
    #increasing = min(increasing, sympy.is_increasing(expr, sympy.Interval(turning_points[-1], interval.end, false, false), var))
    return bool(increasing)

# no parameters to fill variant
def thermo(expr, var):

    if type(expr) == str:
        expr = sympy.parse_expr(expr)

    #println("expression:", expr)
    results = [True, True, True]

    
    # Axiom 1: the expr needs to pass through the origin
    try:
        if sympy.limit(expr, var, 0, "+") != 0:
            #println("constraint 1")
            results[0] = False
    except:
        #println(error)
        #println("SymPy cannot evaluate Axiom 1")
        results[0] = False
    # Axiom 2: the expr needs to converge to Henry's Law at zero pressure
    try:
        if (sympy.limit(sympy.diff(expr, var), var, 0) == sympy.oo 
            or sympy.limit(sympy.diff(expr, var), var, 0) == -sympy.oo 
            or sympy.limit(sympy.diff(expr, var), var, 0) == 0):
            #println("constraint 2")
            results[1] = False
    except:
        #println(error)
        #println("SymPy cannot evaluate Axiom 2")
        results[1] = False

    # Axiom 3: the expr must be strictly increasing as pressure increases
    try:
        # use custom function because sympy doesn't work as expected
        if not(is_monotonic_increasing(expr, sympy.Interval(0,sympy.oo), var)):
            #println("constraint 3")
            results[2] = False
    except:
        #print("SymPy cannot evaluate Axiom 3")
        print("Oops!", sys.exc_info()[0], "occurred for", expr)
        results[2] = False

    #print(results)
    return results

def addThermoVals(run):
    #print("there are", len(run) ,"expressions to check in this run")
    # add thermo pass or fail for normal equations
    p = sympy.symbols("p")
    thermoVals = []
    for expr in run["equation"]:
        thermoVals.append(thermo(sympy.parse_expr(expr), p))
    run[["thermo1", "thermo2", "thermo3"]] = thermoVals

    return run

def main(argv):
    inputFile = ""
    outputFile = ""

    if len(argv) != 2:
        print("Requires input and output file names (pandas csv)")
        sys.exit(2)

    inputFile = argv[0]
    outputFile = argv[1]

    print("reading from", inputFile)

    # do stuff

    # read in parsed data
    data = pd.read_csv(inputFile, index_col=0)
    print(data.columns)
    print(len(data))

    # count number of runs (+1 cause start from 0)
    numRuns = int(data["run"].max() + 1)
    # separate runs for ease of plotting
    runs = []
    for i in range(numRuns):
        thisRun = data.loc[data["run"] == i]
        runs.append(thisRun)
        print("run", i, "length =", len(thisRun))

    #print(runs[0].isnull().sum().sum())
    print(runs[0].head(10))

    for i, run in enumerate(runs):
        print("adding canonical form for run", i+1)
        run = addCanonical(run)

    for i in range(len(runs)):
        print("removing repeats for run", i+1)
        runs[i] = removeRepeats(runs[i])

    for i, run in enumerate(runs):
        print("adding min complexity for run", i+1)
        run["minComplexity"] = run[["complexity", "canonComplexity"]].min(axis=1)

    for i in range(len(runs)):
        print("adding thermo vals for run", i+1)
        runs[i] = addThermoVals(runs[i])

    print("writing to", outputFile)

    if len(runs) > 1:
        data = runs[0].append(runs[1])
        for i in range(2, len(runs)):
            #print(len(data))
            data = data.append(runs[i])
    else:
        data = runs[0]
    data.to_csv(outputFile)

if __name__ == "__main__":
   main(sys.argv[1:])