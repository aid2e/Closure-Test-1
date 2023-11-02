import json
import os
import sys

def ReadJsonFile(jsonFile):
    if(os.path.isfile(jsonFile) == False):
        print ("ERROR: the json file you specified does not exist")
        sys.exit(1)
    with open(jsonFile) as f:
        data = json.loads(f.read())
    return data

def GetDesignParamNames(dataDict, rangeDict):
    designParams = {}
    for key, value in dataDict.items():
        for i in range(1, value[0] + 1):
            key1 = key.replace("_fill_", f"{i}")
            if(rangeDict.get(key1)):
                designParams[key1] = rangeDict[key1]
            else:
                designParams[key1] = rangeDict[key]
    return designParams
