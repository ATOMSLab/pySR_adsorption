# pySR_adsorption

Includes code for running PySR from the commandline, parsing the raw results into a pandas dataframe, adding more columns to that dataframe and finally plotting.  
You will need both Julia and Python to use PySR (and it can be used from either) but you only need Python to handle the output.  Basic order of use:

adsorption.jl -> parsePySRRaw.py -> addCols.py -> visualization
