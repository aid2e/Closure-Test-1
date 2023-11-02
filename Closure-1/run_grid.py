import os, sys

template_config = sys.argv[1]

M = [2, 3, 4]
d = [5, 10, 50, 100]

for ob in M:
    for de in d:
        config = open(template_config, 'r').read()
        config = config.replace('__M__', str(ob))
        config = config.replace('__d__', str(de))
        with open(f"optimize_M{ob}_d{de}.config", "w") as f:
            f.write(config)
        os.system(f"python wrapper.py -c optimize_M{ob}_d{de}.config -p True")