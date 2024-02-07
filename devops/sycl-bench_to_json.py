import subprocess
import sys

result = subprocess.run(sys.argv[1:], stdout=subprocess.PIPE)
if result.returncode == 0:
    names = []
    runtime_mean = []
    runtime_stddev = []
    

    for line in result.stdout.decode("utf-8").splitlines():
        if line.startswith("****"):
            # The name of the benchmark is always printed as ***** name*****
            names.append([s for s in line.split("*") if s != ''][0].strip())

        if line.startswith("run-time-mean"):
            runtime_mean.append(line.split(" ")[1])

        if line.startswith("run-time-stddev"):
            runtime_stddev.append(line.split(" ")[1])
            
    assert(len(names) == len(runtime_mean))
    assert(len(names) == len(runtime_stddev))

    jsonstring = ""
    jsonstring += "["

    for entry in zip(names, runtime_mean, runtime_stddev):
        jsonstring += "{"

        jsonstring += "\"name\":"
        jsonstring += "\"" + entry[0] + "\""
        jsonstring += ","

        jsonstring += "\"unit\":\"s\","

        jsonstring += "\"value\":"
        jsonstring += entry[1]
        jsonstring += ","

        jsonstring += "\"range\":"
        jsonstring += entry[2]

        jsonstring += "},"

    jsonstring = jsonstring[:-1]
    jsonstring += "]"

    print(jsonstring)
