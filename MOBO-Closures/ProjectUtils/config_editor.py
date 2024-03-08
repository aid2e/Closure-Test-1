import json
import os
import sys


def read_json_file(name):
    if not os.path.isfile(name):
        raise FileNotFoundError(f"The specified JSON file ({name}) does not exist.")
    with open(name, 'r') as f:
        return json.load(f)


def GetDesignParamNames(dataDict, rangeDict):
    designParams = {}
    for key, value in dataDict.items():
        for i in range(1, value[0] + 1):
            key1 = key.replace("_fill_", f"{i}")
            if (rangeDict.get(key1)):
                designParams[key1] = rangeDict[key1]
            else:
                designParams[key1] = rangeDict[key]
    return designParams
