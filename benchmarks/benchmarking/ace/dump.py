import sys
import datetime

if __name__=="__main__":
    stat_base = dict(
        log_filename = sys.argv[1],
        solver = sys.argv[2],
        version = sys.argv[3],
        cores = sys.argv[4],
        threads = sys.argv[5],
        timeout = int(sys.argv[6])/1000,
        memout = sys.argv[7],
        input = sys.argv[8],
        extras = sys.argv[9:],
        datetime = datetime.datetime.now().isoformat(),
        experiment_ware = f"{sys.argv[2]}_{sys.argv[3]}"
    )
    
    
    with open(sys.argv[1], "a") as file:
        for k,v in stat_base.items():
            if isinstance(v, list):
                file.write(f"{k}: {' '.join(v)}\n")
            else:
                file.write(f"{k}: {v}\n")
        for line in sys.stdin:
            file.write(line)
            file.flush()
